/* Winding fitter host side (src/fit.h): label-free prior per z knot (pitch from the spacing field, refined by the phase
   coherence of r / pitch - dir theta / 2 pi; offset = its argument), then Adam on delta (coarse to fine) and the knots. */
#include "fit.h"
#include "sysinfo.h"
#include "fit_terms.h"
#include "nn.h"
#include "json.h"
#include "zarr3.h"
#include "store.h"
#include "coreline.h"
#include "l1pot.h"
#include "fit_emb.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/file.h>
#include <fcntl.h>
#include <stdatomic.h>
static double nn_wall_seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int fit_scale_L(int L, double f) { const int v = (int)lround(L * f); return v < 2 ? 2 : v; }
fit_cfg fit_cfg_default(void) {
    fit_cfg c = {0};
    c.dir = 0; c.knot_dz = 1024; c.um = 2.4;
    c.nstage = 6; c.phase_stage = 4;
    const int lat[6] = {512, 256, 128, 64, 64, 32}, st[6] = {300, 300, 300, 300, 300, 300};
    memcpy(c.lattice, lat, sizeof lat); memcpy(c.steps, st, sizeof st);
    c.lr = 0.02f; c.lr_knot = 0.005f;
    c.w_phase = 1.f; c.w_normal = 3.f; c.w_density = 1.f;   /* w_normal 3: 15-24% fewer sheet switches on 3 Paris 4 crops at equal distance (2026-10-19) */ c.w_nondeg = 1.f; c.w_smooth = 0.1f; c.w_decay = 1e-4f;
    c.phase_ramp = 0.1f; c.core = 3.f; c.unwrap_voxels = 1; c.uw_tile = 256; c.analytic = 1; c.batch = 2000000; c.conf_pow = 1.f; c.pgrad_delta = 0.25f;
    c.w_wind4 = 0.f;    /* the winding mod 4 term, active only with dz dy dx dq planes (task unwrap2 fields); off: weight 4 cost fit-val 2026-10-20, 0.5-1 neutral */
    c.w_recto = 0.f;    /* recto anchors off by default */
    c.w_ct = 0.f; c.ct_pow = 1.f; c.ct_phi = 0.f; c.ct_radius_um = 150; c.ct_floor = 0.f;   /* CT parity off by default */
    c.w_cspacing = 0.f; c.cs_tol = 0.25f; c.ct_flood = 0;
    c.seed_z0 = -1e30;
    c.uw_repair = 0; c.uw_rep_qpow = 1.f; c.uw_rep_sr = 1; c.uw_rep_seed = 0.f; c.uw_rep_gd = 8; c.phase_stretch_r = 1;   /* local slip repair off by default */
    c.emb = nullptr; c.uw_rep_emb = 0.f; c.uw_rep_emb_m0 = 0.3f; c.emb_phi = 0.f; c.uw_rep_emb_rmax = 2; c.uw_rep_emb_sub = 2;   /* embedding evidence off */
    c.uw_rep_emb_nd = 2; c.uw_rep_emb_d[0] = 16; c.uw_rep_emb_d[1] = 48; c.uw_rep_emb_d[2] = 128;   /* emb-calib (HOLD): P(r right) 0.99 at conf >= 0.2 for D <= 48, 0.88 at 128 */
    return c;
}

static double axis_lerp(const float *v, int n, double z) {
    if (z <= 0) return v[0];
    if (z >= n - 1) return v[n - 1];
    const int i = (int)z; const double t = z - i;
    return v[i] + t * (v[i + 1] - v[i]);
}
static double knot_lerp(const float *v, int nk, int dz, double z) {
    double k = z / dz; if (k < 0) k = 0; if (k > nk - 1) k = nk - 1;
    int k0 = (int)k; if (k0 > nk - 2) k0 = nk - 2 < 0 ? 0 : nk - 2;
    const double t = nk > 1 ? k - k0 : 0;
    return v[k0] + t * (v[nk > 1 ? k0 + 1 : k0] - v[k0]);
}
static double lattice_at(const float *d, int L, int lz, int ly, int lx, double z, double y, double x) {
    const double q[3] = {z / L, y / L, x / L}; const int dim[3] = {lz, ly, lx}; int c0[3]; double ct[3];
    for (int k = 0; k < 3; k++) {
        double a = q[k] < 0 ? 0 : q[k] > dim[k] - 1 ? dim[k] - 1 : q[k];
        c0[k] = (int)a; if (c0[k] > dim[k] - 2) c0[k] = dim[k] - 2 < 0 ? 0 : dim[k] - 2;
        ct[k] = dim[k] > 1 ? a - c0[k] : 0;
    }
    double s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = c0[0] + (c >> 2 & 1), b = c0[1] + (c >> 1 & 1), e = c0[2] + (c & 1);
        const double w = (c >> 2 & 1 ? ct[0] : 1 - ct[0]) * (c >> 1 & 1 ? ct[1] : 1 - ct[1]) * (c & 1 ? ct[2] : 1 - ct[2]);
        if (w != 0) s += w * d[((size_t)(a < lz ? a : lz - 1) * ly + (b < ly ? b : ly - 1)) * lx + (e < lx ? e : lx - 1)];
    }
    return s;
}
static double fit_base(const fit_model *m, double z, double y, double x, double *th) {   /* r / pitch + offset; theta */
    const int na = (int)(m->D * m->h) + 1;
    const double cy = axis_lerp(m->cy, na, z), cx = axis_lerp(m->cx, na, z);
    *th = atan2(y - cy, x - cx);
    return hypot(y - cy, x - cx) / knot_lerp(m->pitch, m->nk, m->knot_dz, z) + knot_lerp(m->offset, m->nk, m->knot_dz, z);
}
double fit_W(const fit_model *m, double z, double y, double x) {
    double th; const double U = fit_base(m, z, y, x, &th) + lattice_at(m->delta, m->L, m->lz, m->ly, m->lx, z, y, x);
    return U - m->dir * th / (2 * M_PI);
}
/* the same at a point of a ray from the axis: its radius r and angle th (in (-pi, pi]) known, no hypot / atan2 (half of ufsm wraps) */
static double fit_W_rt(const fit_model *m, double z, double y, double x, double r, double th) {
    const double U = r / knot_lerp(m->pitch, m->nk, m->knot_dz, z) + knot_lerp(m->offset, m->nk, m->knot_dz, z) + lattice_at(m->delta, m->L, m->lz, m->ly, m->lx, z, y, x);
    return U - m->dir * th / (2 * M_PI);
}
/* ev_set (and fit_W) along one ray / row: z is fixed, so the slab choice, the blend weight, the knots and the lattice's z interpolation are computed
   once per ray (the same expressions: bit-identical W), the y / x part per evaluation */
typedef struct { const fit_model *m; double zl, pit, off; int c0z; double ctz; size_t za[2]; double iL; } ev_part;   /* za: the two z planes' offsets */
static void ev_part_init(ev_part *p, const fit_model *m, double zl) {
    p->m = m; p->zl = zl; p->pit = knot_lerp(m->pitch, m->nk, m->knot_dz, zl); p->off = knot_lerp(m->offset, m->nk, m->knot_dz, zl);
    const double q = zl / m->L; const double a = q < 0 ? 0 : q > m->lz - 1 ? m->lz - 1 : q;
    p->c0z = (int)a; if (p->c0z > m->lz - 2) p->c0z = m->lz - 2 < 0 ? 0 : m->lz - 2;
    p->ctz = m->lz > 1 ? a - p->c0z : 0;
    for (int b = 0; b < 2; b++) { const int a_ = p->c0z + b; p->za[b] = (size_t)(a_ < m->lz ? a_ : m->lz - 1) * m->ly; }
    p->iL = (m->L & (m->L - 1)) == 0 ? 1.0 / m->L : 0;   /* a power of two: x / L == x * (1 / L) exactly */
}
static inline double ev_part_U(const ev_part *p, double y, double x, double r) {   /* fit_W_rt at (zl, y, x) before its angle term: lattice_at's operations */
    const fit_model *m = p->m; const int L = m->L, ly = m->ly, lx = m->lx;
    const double qy = p->iL > 0 ? y * p->iL : y / L, qx = p->iL > 0 ? x * p->iL : x / L;
    const double ay = qy < 0 ? 0 : qy > ly - 1 ? ly - 1 : qy, ax = qx < 0 ? 0 : qx > lx - 1 ? lx - 1 : qx;
    int c0y = (int)ay; if (c0y > ly - 2) c0y = ly - 2 < 0 ? 0 : ly - 2;
    int c0x = (int)ax; if (c0x > lx - 2) c0x = lx - 2 < 0 ? 0 : lx - 2;
    const double cty = ly > 1 ? ay - c0y : 0, ctx = lx > 1 ? ax - c0x : 0;
    const size_t yo[2] = {(size_t)(c0y < ly ? c0y : ly - 1), (size_t)(c0y + 1 < ly ? c0y + 1 : ly - 1)}, xo[2] = {(size_t)(c0x < lx ? c0x : lx - 1), (size_t)(c0x + 1 < lx ? c0x + 1 : lx - 1)};
    const double wz[2] = {1 - p->ctz, p->ctz}, wy[2] = {1 - cty, cty}, wx[2] = {1 - ctx, ctx};
    double s = 0;
    for (int c = 0; c < 8; c++) {   /* corner order and weight products as lattice_at: (wz wy) wx */
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        const double w = wz[bz] * wy[by] * wx[bx], t = w * m->delta[(p->za[bz] + yo[by]) * lx + xo[bx]];
        s += w != 0 ? t : 0.0;   /* (a select, not a branch: s starts at +0 and is never -0, so adding +0 for a skipped corner is exact) */
    }
    return r / p->pit + p->off + s;
}
static double ev_part_W(const ev_part *p, double y, double x, double r, double th) {   /* fit_W_rt at (zl, y, x) */
    return ev_part_U(p, y, x, r) - p->m->dir * th / (2 * M_PI);
}
/* ev_part at one y as well (a field row: the voxel loops' x runs): the four (z, y) corners' offsets and weight products wz wy, so per
   x only its part (the same operations as ev_part_W: bit-identical) */
typedef struct { const ev_part *p; size_t base[4]; double wzy[4]; } ev_party;
static void ev_party_init(ev_party *q, const ev_part *p, double y) {
    const fit_model *m = p->m; const int L = m->L, ly = m->ly;
    const double qy = p->iL > 0 ? y * p->iL : y / L, ay = qy < 0 ? 0 : qy > ly - 1 ? ly - 1 : qy;
    int c0y = (int)ay; if (c0y > ly - 2) c0y = ly - 2 < 0 ? 0 : ly - 2;
    const double cty = ly > 1 ? ay - c0y : 0, wz[2] = {1 - p->ctz, p->ctz}, wy[2] = {1 - cty, cty};
    const size_t yo[2] = {(size_t)(c0y < ly ? c0y : ly - 1), (size_t)(c0y + 1 < ly ? c0y + 1 : ly - 1)};
    for (int c = 0; c < 4; c++) { q->wzy[c] = wz[c >> 1] * wy[c & 1]; q->base[c] = (p->za[c >> 1] + yo[c & 1]) * m->lx; }
    q->p = p;
}
static double ev_party_W(const ev_party *q, double x, double r, double th) {   /* ev_part_W at (q's y, x) */
    const ev_part *p = q->p; const fit_model *m = p->m; const int L = m->L, lx = m->lx;
    const double qx = p->iL > 0 ? x * p->iL : x / L, ax = qx < 0 ? 0 : qx > lx - 1 ? lx - 1 : qx;
    int c0x = (int)ax; if (c0x > lx - 2) c0x = lx - 2 < 0 ? 0 : lx - 2;
    const double ctx = lx > 1 ? ax - c0x : 0, wx[2] = {1 - ctx, ctx};
    const size_t xo[2] = {(size_t)(c0x < lx ? c0x : lx - 1), (size_t)(c0x + 1 < lx ? c0x + 1 : lx - 1)};
    double s = 0;
    for (int c = 0; c < 8; c++) { const double w = q->wzy[c >> 1] * wx[c & 1]; if (w != 0) s += w * m->delta[q->base[c >> 1] + xo[c & 1]]; }
    const double U = r / p->pit + p->off + s;
    return U - m->dir * th / (2 * M_PI);
}

static int cmp_f(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

/* Label-free prior. Per knot (voxels within one knot spacing in z): pitch0 = the median sheet spacing, refined in [0.7, 1.4]
   pitch0 by the incoherent sum over angular sectors and radial rings of |sum w m e^{-i 2 pi r / p}| (robust to angular
   modulation of the sheets). Direction: per radial ring the sector offsets arg(sum w m e^{-i 2 pi r / p}) / 2 pi, unwrapped
   around the circle, change by -dir in total (m = e^{i 2 pi W}, W = r / p - dir theta / 2 pi + periodic); dir = -sign of the
   median total. Offset per knot: arg sum w m e^{-i 2 pi (r / p - dir theta / 2 pi)} / 2 pi, unwrapped across knots. */
#define FIT_NSEC 64
typedef struct { double r, th, w; double complex m; } fit_sample;
static double sector_coherence(const fit_sample *v, size_t ns, double p, double ring) {
    const int nr = 64; double complex acc[64 * FIT_NSEC]; memset(acc, 0, sizeof acc);
    for (size_t i = 0; i < ns; i++) {
        int ri = (int)(v[i].r / ring); if (ri >= nr) ri = nr - 1;
        int si = (int)((v[i].th + M_PI) / (2 * M_PI) * FIT_NSEC); if (si >= FIT_NSEC) si = FIT_NSEC - 1;
        acc[ri * FIT_NSEC + si] += v[i].w * v[i].m * cexp(-I * 2 * M_PI * v[i].r / p);
    }
    double c = 0; for (int k = 0; k < nr * FIT_NSEC; k++) c += cabs(acc[k]);
    return c;
}
static int ring_winding(const fit_sample *v, size_t ns, double p, double ring, double *totals, int maxr) {   /* per ring: unwrapped total */
    const int nr = maxr; double complex *acc = calloc((size_t)nr * FIT_NSEC, sizeof *acc); double *wsum = calloc((size_t)nr * FIT_NSEC, sizeof(double));
    for (size_t i = 0; i < ns; i++) {
        int ri = (int)(v[i].r / ring); if (ri >= nr) continue;
        int si = (int)((v[i].th + M_PI) / (2 * M_PI) * FIT_NSEC); if (si >= FIT_NSEC) si = FIT_NSEC - 1;
        acc[ri * FIT_NSEC + si] += v[i].w * v[i].m * cexp(-I * 2 * M_PI * v[i].r / p); wsum[ri * FIT_NSEC + si] += v[i].w;
    }
    int nt = 0;
    for (int ri = 0; ri < nr; ri++) {
        int have = 0, first = -1; double prev = 0, tot = 0, firstv = 0;
        for (int si = 0; si < FIT_NSEC; si++) {
            const double complex a = acc[ri * FIT_NSEC + si];
            if (wsum[ri * FIT_NSEC + si] <= 0 || cabs(a) < 0.2 * wsum[ri * FIT_NSEC + si]) continue;   /* incoherent sector */
            const double ph = carg(a) / (2 * M_PI);
            if (first < 0) { first = si; firstv = prev = ph; }
            else { double d = ph - prev; d -= round(d); tot += d; prev = ph; }
            have++;
        }
        if (have < FIT_NSEC * 3 / 4) continue;
        double d = firstv - prev; d -= round(d); tot += d;   /* close the circle */
        totals[nt++] = tot;
    }
    free(acc); free(wsum);
    return nt;
}
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
/* the prior's phase vector of voxel i: the network phase, plus with recto anchors (w_recto > 0, an r plane) w_recto r^2 at phase zero
   (recto peaks sit on integer W, independently of the phase head) */
static double complex prior_phasor(const fit_fields *f, const fit_cfg *c, size_t i) {
    const double complex phase = (f->pc[i] - 128) / 127.0 + I * ((f->ps[i] - 128) / 127.0);
    if (!f->recto || !(c->w_recto > 0)) return phase;
    const double r = f->recto[i] / 255.0;
    return c->w_phase * phase + c->w_recto * r * r;
}
typedef struct { const fit_sample *v; size_t ns; double p0; double *cc; } coh_job;
static void fit_zpar(int D, int nt, void (*fn)(void *, int, int, int), void *ctx);
static int fit_nthreads(int D);
static void coh_run(void *a, int s0, int s1, int t_) { const coh_job *J = a; for (int s = s0; s < s1; s++) J->cc[s] = sector_coherence(J->v, J->ns, J->p0 * (0.7 + 0.01 * s), 4 * J->p0); }
int fit_init_prior(const fit_fields *f, const fit_cfg *c, fit_model *m) {
    const double h = f->h; const size_t n = (size_t)f->D * f->H * f->W;
    const int na = (int)(f->D * h) + 1;
    m->knot_dz = c->knot_dz; m->nk = (int)ceil(f->D * h / c->knot_dz) + 1; if (m->nk < 2) m->nk = 2;
    m->pitch = calloc(m->nk, sizeof(float)); m->offset = calloc(m->nk, sizeof(float));
    const size_t stride = n > 400000 ? n / 400000 : 1;
    double *pk = calloc(m->nk, sizeof(double)), *ok = calloc(m->nk, sizeof(double)), wind = 0; int nwind = 0;
    double *tots = calloc((size_t)m->nk * 256, sizeof(double)); int ntot = 0;
    fit_sample *v = malloc((n / stride + 1) * sizeof *v);
    float *sps = malloc((n / stride + 1) * sizeof(float));
    for (int k = 0; k < m->nk; k++) {
        const double zc = (double)k * c->knot_dz, zlo = zc - c->knot_dz, zhi = zc + c->knot_dz;
        size_t ns = 0, nsp = 0;
        for (size_t i = 0; i < n; i += stride) {
            const int z = (int)(i / ((size_t)f->H * f->W)), y = (int)((i / f->W) % f->H), x = (int)(i % f->W); const double pz = (z + 0.5) * h;
            if (pz < zlo || pz > zhi || !f->w[i]) continue;
            if (f->sp[i]) sps[nsp++] = f->sp[i] * (float)h / 8.f;
            const double complex phasor = prior_phasor(f, c, i); const double mc = creal(phasor), ms = cimag(phasor);
            if (mc * mc + ms * ms < 0.01) continue;
            const double cy = axis_lerp(f->cy, na, pz), cx = axis_lerp(f->cx, na, pz), py = (y + 0.5) * h, px = (x + 0.5) * h;
            v[ns++] = (fit_sample){hypot(py - cy, px - cx), atan2(py - cy, px - cx), f->w[i] / 255.0, mc + I * ms};
        }
        const int gonly = c->phase_stage >= c->nstage;   /* a gradient-only fit (fit --coarse-pool: pooled phase carries nothing): */
        if (!nsp || (!ns && !gonly)) { pk[k] = -1; continue; }
        qsort(sps, nsp, sizeof(float), cmp_f);
        const double p0 = sps[nsp / 2];
        if (!ns) { pk[k] = p0; continue; }   /* the pitch from the spacing alone (offset 0: the gradient terms set no gauge) */
        double bestc = -1, bestp = p0, cc[71];
        { coh_job J = {v, ns, p0, cc}; fit_zpar(71, fit_nthreads(71), coh_run, &J); }   /* the 71 trial pitches on threads */
        for (int s = 0; s <= 70; s++) { const double p = p0 * (0.7 + 0.01 * s); if (cc[s] > bestc) { bestc = cc[s]; bestp = p; } }
        pk[k] = bestp;
        ntot += ring_winding(v, ns, bestp, 3 * bestp, tots + ntot, 256 < (int)((size_t)m->nk * 256 - ntot) ? 256 : (int)((size_t)m->nk * 256 - ntot));
        /* keep this knot's samples' offset for the chosen direction below: store per knot later */
    }
    if (ntot) { qsort(tots, ntot, sizeof(double), cmp_d); wind = tots[ntot / 2]; nwind = ntot; }
    m->dir = c->dir ? c->dir : (wind > 0 ? -1 : 1);
    m->dir_vote = c->dir ? 1.0 : fabs(wind) * (nwind >= 3 ? 1 : 0);   /* |median winding per ring|, 0 when too few rings */
    for (int k = 0; k < m->nk; k++) {   /* offset per knot with the direction */
        if (pk[k] <= 0) continue;
        const double zc = (double)k * c->knot_dz, zlo = zc - c->knot_dz, zhi = zc + c->knot_dz;
        double complex acc = 0;
        for (size_t i = 0; i < n; i += stride) {
            const int z = (int)(i / ((size_t)f->H * f->W)), y = (int)((i / f->W) % f->H), x = (int)(i % f->W); const double pz = (z + 0.5) * h;
            if (pz < zlo || pz > zhi || !f->w[i]) continue;
            const double complex phasor = prior_phasor(f, c, i); const double mc = creal(phasor), ms = cimag(phasor);
            const double cy = axis_lerp(f->cy, na, pz), cx = axis_lerp(f->cx, na, pz), py = (y + 0.5) * h, px = (x + 0.5) * h;
            const double r = hypot(py - cy, px - cx), th = atan2(py - cy, px - cx);
            acc += f->w[i] / 255.0 * (mc + I * ms) * cexp(-I * 2 * M_PI * (r / pk[k] - m->dir * th / (2 * M_PI)));
        }
        ok[k] = carg(acc) / (2 * M_PI);
    }
    free(sps); free(v); free(tots);
    int first = -1;
    for (int k = 0; k < m->nk; k++) if (pk[k] > 0) { first = k; break; }
    if (first < 0) { free(pk); free(ok); fprintf(stderr, "fit: no data for the prior\n"); return -1; }
    double prev_o = ok[first], prev_p = pk[first];
    for (int k = 0; k < m->nk; k++) {
        double p = pk[k] > 0 ? pk[k] : prev_p, o = pk[k] > 0 ? ok[k] : prev_o;
        o += round(prev_o - o);   /* continuous across knots */
        m->pitch[k] = (float)p; m->offset[k] = (float)o; prev_o = o; prev_p = p;
    }
    if (c->verbose) fprintf(stderr, "fit prior: dir %+d (median winding per ring %.3f over %d rings), pitch %.2f..%.2f over %d knots\n", m->dir, wind, nwind, m->pitch[0], m->pitch[m->nk - 1], m->nk);
    free(pk); free(ok);
    return 0;
}

typedef struct { const fit_model *m; int L, lz, ly, lx; const fit_cfg *c; float *d; } prolong_job;
static void prolong_planes(void *a_, int a0, int a1, int t_) {
    const prolong_job *J = a_; const fit_model *m = J->m; const int L = J->L, ly = J->ly, lx = J->lx; const fit_cfg *c = J->c; float *d = J->d;
    if (m->delta) for (int a = a0; a < a1; a++) for (int b = 0; b < ly; b++) for (int e = 0; e < lx; e++)
        d[((size_t)a * ly + b) * lx + e] = (float)lattice_at(m->delta, m->L, m->lz, m->ly, m->lx, (double)a * L, (double)b * L, (double)e * L);
    else if (c && c->init) {   /* the first stage from a given model: delta = its U minus this model's base at the nodes, so that U
                                 starts equal to the model's (knots interpolated differently, axis tables of different extent) */
        const fit_model *g = c->init; double th;
        for (int a = a0; a < a1; a++) for (int b = 0; b < ly; b++) for (int e = 0; e < lx; e++) {
            const double z = (double)a * L, y = (double)b * L, x = (double)e * L, zg = z + c->init_off[0], yg = y + c->init_off[1], xg = x + c->init_off[2];
            const double Ug = fit_base(g, zg, yg, xg, &th) + lattice_at(g->delta, g->L, g->lz, g->ly, g->lx, zg, yg, xg);
            d[((size_t)a * ly + b) * lx + e] = (float)(Ug - fit_base(m, z, y, x, &th));
        }
    }
}
static float *prolong(const fit_model *m, int L, int lz, int ly, int lx, const fit_cfg *c) {   /* old delta at the new lattice nodes (planes on threads) */
    float *d = calloc((size_t)lz * ly * lx, sizeof(float));
    prolong_job J = {m, L, lz, ly, lx, c, d};
    if (m->delta || (c && c->init)) fit_zpar(lz, fit_nthreads(lz), prolong_planes, &J);
    return d;
}

/* Phase-unwrapping initialisation at the phase stage: per lattice node the weighted residual phasor sum of the voxels around it,
   A_n = sum w conf e^{i (phi - 2 pi W)}; corrections arg(A_n) / 2 pi unwrapped node to node in order of coherence |A_n| / sum w
   (each new node takes the integer that keeps it within half a turn of the visited neighbour), nodes below min_coh filled by
   averaging their visited neighbours; delta += correction. The gradient stages leave W with the right local spacing but a drift
   of possibly several tenths of a turn; this puts every region on the phase's integer before the periodic phase term starts. */
typedef struct { double c; int i; } fit_heap_e;
static void heap_push(fit_heap_e *h, int *n, fit_heap_e e) { int i = (*n)++; h[i] = e; while (i && h[(i - 1) / 2].c < h[i].c) { fit_heap_e t = h[i]; h[i] = h[(i - 1) / 2]; h[(i - 1) / 2] = t; i = (i - 1) / 2; } }
static fit_heap_e heap_pop(fit_heap_e *h, int *n) {
    fit_heap_e top = h[0]; h[0] = h[--(*n)]; int i = 0;
    for (;;) { int l = 2 * i + 1, r = l + 1, b = i; if (l < *n && h[l].c > h[b].c) b = l; if (r < *n && h[r].c > h[b].c) b = r; if (b == i) break; fit_heap_e t = h[i]; h[i] = h[b]; h[b] = t; i = b; }
    return top;
}
/* Threads over field z planes for the voxel loops of the unwrapping: fn(ctx, z0, z1, t) on nt contiguous ranges (UFSM_FIT_THREADS
   overrides the core count) */
typedef void (*fit_zfn)(void *ctx, int z0, int z1, int t);
typedef struct { fit_zfn fn; void *ctx; int z0, z1, t; } fit_zjob;
static void *fit_zrun(void *a) { fit_zjob *j = a; j->fn(j->ctx, j->z0, j->z1, j->t); return nullptr; }
static int g_fit_threads;   /* fit_cfg threads of the current fit (0: the core count) */
void fit_set_threads(int n) { g_fit_threads = n; }
static int fit_nthreads(int D) {
    long n = ufsm_ncpu(); const char *e = getenv("UFSM_FIT_THREADS"); if (e && atoi(e) > 0) n = atoi(e);
    if (g_fit_threads > 0) n = g_fit_threads;
    if (n > 64) n = 64;
    if (n > D) n = D;
    return n < 1 ? 1 : (int)n;
}
static void fit_zpar(int D, int nt, fit_zfn fn, void *ctx) {
    pthread_t th[64]; fit_zjob jb[64]; int up[64] = {0};
    for (int t = 0; t < nt; t++) jb[t] = (fit_zjob){fn, ctx, (int)((long)D * t / nt), (int)((long)D * (t + 1) / nt), t};
    for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, fit_zrun, &jb[t]) == 0;
    fit_zrun(&jb[0]);
    for (int t = 1; t < nt; t++) { if (up[t]) pthread_join(th[t], nullptr); else fit_zrun(&jb[t]); }
}
int fit_par_threads(int n) { return fit_nthreads(n); }
int fit_par(int n, void (*fn)(void *ctx, int a0, int a1, int t), void *ctx) { if (n <= 0) return 0; const int nt = fit_nthreads(n); fit_zpar(n, nt, fn, ctx); return nt; }

/* Voxel-level quality-guided phase unwrapping: per field voxel the wrapped unwrapped-radius u = frac(phi + dir theta / 2 pi) (U mod 1,
   single-valued away from the axis), flooded from the most confident voxel in order of confidence (phasor length x material),
   each new voxel taking the integer that keeps it within half a turn of the voxel it is reached from (6-neighbours; one field
   voxel is ~0.1 turn); the seed takes the integer nearest the current model. Then per lattice node the mean of (U_unwrapped -
   U_model) over its voxels (no integer ambiguity left) is added to delta; nodes without voxels get their neighbours' mean.
   Speed (a 3.9 cm x 2.5 mm slab at 19.2 um: 537M field voxels, 242M flooded): the flood is one serial walk bound by memory latency
   (~10 cache lines per voxel), 250 s with the binary heap it replaces -> 30 s: 8 B records (float, 16-bit integer relative to the
   seed's, 16-bit level; 16 B: 36 -> 47 s) in 8^3 bricks (4^3: 35 s at prefetch 8) on huge pages, a bucket queue (confidence in
   4094 levels, FIFO within a level, the next non-empty level from a two-level bitmap) of per-level id arrays, so the voxel 16
   pops ahead is known and its record and z / y neighbours are prefetched (none: 57 s; 8 / 32 ahead: 36 / 33 s). The result is
   the original's (same voxels, same integers). Tiled floods stitched by votes were 30x faster but got 14-16% of the winding ids
   against 19.6% (each tile commits its own low-confidence bridges). The voxel loops run on threads. Returns the number of voxels
   reached. The integers do not depend on the seed (up to the gauge): the quality-ordered flood is Prim's maximum spanning tree of
   the voxel-quality graph, whatever its root (2026-10-19: 5 floods from seeds spread over a Paris 4 crop differed on 0 voxels), so
   consensus over seeds gains nothing; only the quality order (uw_q) shapes where the flood crosses dislocations. */
enum { UQ_LEVELS = 4096 };
#define UQ_NONE UINT32_MAX
#define UK_UNSET 0         /* an all-zero record: not reached, bk 0 (the records start as fresh zero pages: only the bricks holding
                              unwrappable voxels take memory) */
#define UK_BIAS 16384      /* the integer base kb sits this far below the seed's integer, so reached voxels hold k > 0 (k in 1..32767:
                              -16383..+16383 turns from the seed) */
typedef struct { float uw; int16_t k; uint16_t bk; } uw_vox;   /* wrapped U; integer turns minus the base kb (UK_UNSET: not reached);
                                                                confidence bucket (0: not unwrapped). 8 B: the flood is memory bound */
static_assert(sizeof(uw_vox) == 8);
/* records in UB_S^3 bricks (UB = log2 UB_S; a voxel's six neighbours mostly within the brick; linear rows put z neighbours 32 MB away) */
#ifndef UB
#define UB 3
#endif
enum { UB_S = 1 << UB, UB_M = UB_S - 1 };
typedef struct { int D, H, W, bH, bW; } uw_geo;
static inline uint32_t ub_id(const uw_geo *g, int z, int y, int x) {
    return (uint32_t)(((((size_t)(z >> UB) * g->bH + (y >> UB)) * g->bW + (x >> UB)) << (3 * UB)) | (uint32_t)((z & UB_M) << (2 * UB) | (y & UB_M) << UB | (x & UB_M)));
}
static inline void ub_zyx(const uw_geo *g, uint32_t id, int *z, int *y, int *x) {
    const uint32_t b = id >> (3 * UB), r = id & ((1u << (3 * UB)) - 1), bx = b % g->bW, t = b / g->bW;
    *z = (int)((t / g->bH) << UB | r >> (2 * UB)); *y = (int)((t % g->bH) << UB | (r >> UB & UB_M)); *x = (int)(bx << UB | (r & UB_M));
}
typedef struct {
    fit_model *m; const fit_fields *f; const fit_cfg *c; double min_conf; int na, nt; uw_geo g;
    uw_vox *V; int32_t kb;                 /* the records; their integer base (the seed's integer) */
    _Atomic size_t over;                   /* integers more than 16383 turns from the seed (left unreached) */
    float qb[64]; size_t sb[64];            /* per thread: the best confidence and its voxel */
    double *acc, *wsum;
    double rbias;                          /* fit_cfg ref: the sub-turn offset U_ref - u (turns) */
    double complex rph[64]; double rpw[64]; /* per thread: its phasor sums */
    size_t nreach[64];                     /* per thread: the voxels uw_refsnap gave an integer */
} uw_ctx;
/* the ref model's (fit_cfg ref) U at a field voxel: W_ref at the same point of its box plus dir theta / 2 pi about this box's axis */
static double uw_Uref(const uw_ctx *x, int z, int y, int xx) {
    const fit_fields *f = x->f; const fit_cfg *c = x->c; const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
    const double th = atan2(py - axis_lerp(f->cy, x->na, pz), px - axis_lerp(f->cx, x->na, pz));
    return fit_set_W(c->ref, pz + c->ref_off[0], py + c->ref_off[1], px + c->ref_off[2]) + x->m->dir * th / (2 * M_PI);
}
/* the records: fresh anonymous zero pages (= UK_UNSET, bk 0), written only where a voxel can be unwrapped. Up to UFSM_UW_HUGE_GB (default 0: never; huge pages stall in direct compaction when memory is fragmented, 15 s seen on a 10 M-voxel box)
   on transparent huge pages (the flood is memory-latency bound: fewer TLB misses), above that on 4 KB pages (one 8^3 brick per page),
   so a full cross-section (~6% of its voxels reached) takes the bricks with material, not 8 B per voxel; *base / *len: the mapping
   (munmap) */
static void *uw_alloc_records(size_t bytes, void **base, size_t *len) {
    const size_t A = (size_t)2 << 20; bytes = (bytes + A - 1) / A * A;
    void *p = mmap(nullptr, bytes + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    const char *e = getenv("UFSM_UW_HUGE_GB"); const double hg = e ? atof(e) : 0.0;
    uint8_t *a = (uint8_t *)(((uintptr_t)p + A - 1) / A * A);   /* (2 MB aligned inside the mapping) */
    madvise(a, bytes, (double)bytes <= hg * 1e9 ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    *base = p; *len = bytes + A;
    return a;
}
/* phase derivative variance of voxel i (turns^2): the mean square of the wrapped second differences of the phase along z, y and x
   (over the axes whose two neighbours carry a phase); ~0 where the phase is locally linear, large at dislocations and noise */
static double uw_pdv(const fit_fields *f, size_t i, double a0) {
    const size_t HW = (size_t)f->H * f->W, st[3] = {HW, (size_t)f->W, 1};
    const int pos[3] = {(int)(i / HW), (int)((i / f->W) % f->H), (int)(i % f->W)}, dim[3] = {f->D, f->H, f->W};
    double s = 0; int n = 0;
    for (int d = 0; d < 3; d++) {
        if (pos[d] <= 0 || pos[d] >= dim[d] - 1) continue;
        const size_t im = i - st[d], ip = i + st[d];
        const int cm = f->pc[im] - 128, sm = f->ps[im] - 128, cp = f->pc[ip] - 128, sp = f->ps[ip] - 128;
        if (!f->w[im] || !f->w[ip] || cm * cm + sm * sm < 40 || cp * cp + sp * sp < 40) continue;   /* conf < 0.05 */
        double e = (atan2(sp, cp) - 2 * a0 + atan2(sm, cm)) / (2 * M_PI); e -= round(e); s += e * e; n++;
    }
    return n ? s / n : 0;
}
/* atan2(ms, mc) of every (pc, ps) byte pair (uw_q's phase angle: the same expression, looked up) */
static double g_uw_ang[65536]; static pthread_once_t g_uw_ang_once = PTHREAD_ONCE_INIT;
static void uw_ang_init(void) { for (int p = 0; p < 256; p++) for (int s = 0; s < 256; s++) g_uw_ang[p << 8 | s] = atan2((s - 128) / 127.0, (p - 128) / 127.0); }
/* the voxel's confidence (0: excluded) and its phase angle (ang null: not wanted); fit_cfg uw_pdv > 0: times exp(-pdv / uw_pdv^2) */
static float uw_q(const fit_fields *f, size_t i, double min_conf, const fit_cfg *c, double *ang) {
    if (!f->w[i]) return 0;
    const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0, conf = sqrt(mc * mc + ms * ms);
    if (conf < min_conf) return 0;
    if (c->packed_um > 0 && f->sp[i] && f->sp[i] * f->h / 8.0 * (c->um > 0 ? c->um : 2.4) < 0.6 * c->packed_um) return 0;   /* packed: no phase integers */
    const double a = ang || c->uw_pdv > 0 ? g_uw_ang[f->pc[i] << 8 | f->ps[i]] : 0; if (ang) *ang = a;   /* (= atan2(ms, mc)) */
    double q = conf * f->w[i] / 255.0;
    if (c->uw_pdv > 0) q *= exp(-uw_pdv(f, i, a) / (c->uw_pdv * c->uw_pdv));
    return (float)q;
}
static void uw_setup(void *a, int z0, int z1, int t) {
    uw_ctx *x = a; const fit_fields *f = x->f; const size_t HW = (size_t)f->H * f->W;
    float qb = 0; size_t sb = (size_t)-1;
    for (int z = z0; z < z1; z++) {
        const double pz = (z + 0.5) * f->h, cy = axis_lerp(f->cy, x->na, pz), cx = axis_lerp(f->cx, x->na, pz);
        for (int y = 0; y < f->H; y++) for (int xx = 0; xx < f->W; xx++) {
            const size_t i = z * HW + (size_t)y * f->W + xx; double ang = 0;
            const float q = uw_q(f, i, x->min_conf, x->c, &ang);
            if (q <= 0) continue;   /* (the record stays zero: uw 0, UK_UNSET, bk 0; its page untouched) */
            uw_vox *v = &x->V[ub_id(&x->g, z, y, xx)]; v->k = UK_UNSET;
            const double th = atan2((y + 0.5) * f->h - cy, (xx + 0.5) * f->h - cx);
            double u = ang / (2 * M_PI) + x->m->dir * th / (2 * M_PI); u -= floor(u);
            v->uw = (float)u; v->bk = (uint16_t)(1 + (int)(fminf(q, 1.f) * (UQ_LEVELS - 2)));
            if (q > qb) { qb = q; sb = i; }
        }
    }
    x->qb[t] = qb; x->sb[t] = sb;
}
static void uw_seedmodel(void *a, int z0, int z1, int t) {   /* the seed model's region [seed_z0, seed_z): its integers */
    uw_ctx *x = a; const fit_fields *f = x->f; const fit_cfg *c = x->c;
    for (int z = z0; z < z1; z++) {
        const double pz = (z + 0.5) * f->h; if (pz >= c->seed_z || pz < c->seed_z0) continue;
        const double cy = axis_lerp(f->cy, x->na, pz), cx = axis_lerp(f->cx, x->na, pz);
        for (int y = 0; y < f->H; y++) for (int xx = 0; xx < f->W; xx++) {
            uw_vox *v = &x->V[ub_id(&x->g, z, y, xx)]; if (!v->bk) continue;
            const double py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h, th = atan2(py - cy, px - cx);
            const double Us = fit_W(c->seed, pz + c->seed_off, py, px) + x->m->dir * th / (2 * M_PI);
            const int32_t kr = (int32_t)round(Us - v->uw) - x->kb;
            if (kr > UK_UNSET && kr <= INT16_MAX) v->k = (int16_t)kr; else atomic_fetch_add(&x->over, 1);
        }
    }
}
static void uw_refbias(void *a, int z0, int z1, int t) {   /* fit_cfg ref: phasor sums of U_ref - u over every other z row, every 4th y / x */
    uw_ctx *x = a; const fit_fields *f = x->f; double complex s = 0; double w = 0;
    for (int z = z0; z < z1; z++) {
        if (z & 1) continue;
        for (int y = 0; y < f->H; y += 4) for (int xx = 0; xx < f->W; xx += 4) {
            const uw_vox *v = &x->V[ub_id(&x->g, z, y, xx)]; if (!v->bk) continue;
            s += v->bk * cexp(2 * M_PI * I * (uw_Uref(x, z, y, xx) - v->uw)); w += v->bk;
        }
    }
    x->rph[t] = s; x->rpw[t] = w;
}
static void uw_refsnap(void *a, int z0, int z1, int t) {   /* fit_cfg ref_snap: every unwrappable voxel's integer from the ref (uw_Uref; a
                                                              single-model ref per row: ev_part, bit-identical to fit_set_W) */
    uw_ctx *x = a; const fit_fields *f = x->f; const fit_cfg *c = x->c; const fit_set *R = c->ref; const fit_model *G = R->global;
    const int nag = (int)(G->D * G->h) + 1; size_t nr = 0;
    for (int z = z0; z < z1; z++) {
        const double pz = (z + 0.5) * f->h, cy = axis_lerp(f->cy, x->na, pz), cx = axis_lerp(f->cx, x->na, pz), zr = pz + c->ref_off[0];
        ev_part P; double gy = 0, gx = 0; if (!R->n) { ev_part_init(&P, G, zr); gy = axis_lerp(G->cy, nag, zr); gx = axis_lerp(G->cx, nag, zr); }
        for (int y = 0; y < f->H; y++) {
            ev_party Q; if (!R->n) ev_party_init(&Q, &P, (y + 0.5) * f->h + c->ref_off[1]);
            for (int xx = 0; xx < f->W; xx++) {
                uw_vox *v = &x->V[ub_id(&x->g, z, y, xx)]; if (!v->bk) continue;
                const double py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h, th = atan2(py - cy, px - cx), yr = py + c->ref_off[1], xr = px + c->ref_off[2];
                const double Wr = R->n ? fit_set_W(R, zr, yr, xr) : ev_party_W(&Q, xr, hypot(yr - gy, xr - gx), atan2(yr - gy, xr - gx));
                const int32_t kr = (int32_t)round(Wr + x->m->dir * th / (2 * M_PI) - x->rbias - v->uw) - x->kb;
                if (kr > UK_UNSET && kr <= INT16_MAX) { v->k = (int16_t)kr; nr++; } else atomic_fetch_add(&x->over, 1);
            }
        }
    }
    x->nreach[t] = nr;
}
/* per lattice node the q-weighted sum of U_unwrapped - U_model over the reached voxels nearest to it. Work items = node rows (az, bb)
   (lattice z plane, y row; fit_par over lz x ly items: every thread busy, where a thread per z plane left 12 threads 13 planes): the
   item's field rows zr[az]..zr[az + 1) x yr[bb]..yr[bb + 1), so each node's sum runs over its voxels in (z, y, x) order as before. The
   model's W per row (ev_part: knots and lattice z part once, bit-identical to fit_W) with theta computed once per voxel. */
typedef struct { uw_ctx *x; const int *zr, *yr; } ua_job;
static void uw_accum(void *a, int i0, int i1, int t_) {
    const ua_job *J = a; uw_ctx *x = J->x; const fit_fields *f = x->f; const fit_model *m = x->m; const size_t HW = (size_t)f->H * f->W; (void)t_;
    for (int it = i0; it < i1; it++) {
        const int az = it / m->ly, bb = it % m->ly;
        for (int z = J->zr[az]; z < J->zr[az + 1]; z++) {
            const double pz = (z + 0.5) * f->h, cy = axis_lerp(f->cy, x->na, pz), cx = axis_lerp(f->cx, x->na, pz);
            ev_part P; ev_part_init(&P, m, pz);
            for (int y = J->yr[bb]; y < J->yr[bb + 1]; y++) {
                ev_party Q; ev_party_init(&Q, &P, (y + 0.5) * f->h);
                for (int xx = 0; xx < f->W; xx++) {
                    const size_t i = z * HW + (size_t)y * f->W + xx; const uw_vox *v = &x->V[ub_id(&x->g, z, y, xx)]; if (v->k == UK_UNSET) continue;
                    const float q = uw_q(f, i, x->min_conf, x->c, nullptr);
                    const double py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h, dy = py - cy, dx = px - cx, th = atan2(dy, dx);
                    const double Um = ev_party_W(&Q, px, hypot(dy, dx), th) + m->dir * th / (2 * M_PI), Uv = (double)v->uw + (x->kb + v->k);
                    const int cc = (int)lround(px / m->L);
                    const size_t kn = ((size_t)az * m->ly + bb) * m->lx + (cc < m->lx ? cc : m->lx - 1);
                    x->acc[kn] += q * (Uv - Um); x->wsum[kn] += q;
                }
            }
        }
    }
}
/* the bucket queue: per confidence level an array of voxel ids, appended at n and popped at h (FIFO; reset when emptied): sequential
   pushes, and the ids UQ_PD ahead are known, so their records are prefetched long before they are popped */
typedef struct { uint32_t *a; size_t n, h, cap; } uq_lev;
typedef struct { uq_lev *Q; int cur; uint64_t occ[UQ_LEVELS / 64], occ1; } uw_queue;   /* non-empty levels: a bit per level, a bit per
                                                                                          word (the next level down in O(1)) */
static_assert(UQ_LEVELS / 64 <= 64);
enum { UQ_PD = 16 };
static int uq_init(uw_queue *q) { memset(q, 0, sizeof *q); q->Q = calloc(UQ_LEVELS, sizeof *q->Q); return q->Q ? 0 : -1; }
static void uq_free(uw_queue *q) { if (q->Q) for (int b = 0; b < UQ_LEVELS; b++) free(q->Q[b].a); free(q->Q); q->Q = nullptr; }
static inline void uq_push(uw_queue *q, uint32_t j, int b) {
    uq_lev *l = &q->Q[b];
    if (l->n == l->cap) { l->cap = l->cap ? 2 * l->cap : 4096; l->a = realloc(l->a, l->cap * sizeof *l->a); if (!l->a) abort(); }
    l->a[l->n++] = j; q->occ[b >> 6] |= 1ull << (b & 63); q->occ1 |= 1ull << (b >> 6); if (b > q->cur) q->cur = b;
}
static inline int uq_pop(uw_queue *q, uint32_t *i, uint32_t *ahead) {   /* 0: empty; *ahead: the id UQ_PD pops ahead (UQ_NONE: none) */
    while (q->Q[q->cur].h == q->Q[q->cur].n) {   /* level emptied: reset it, the highest marked level next (a level a push left behind
                                                    may be marked yet empty: hence the loop) */
        const int cur = q->cur; q->Q[cur].h = q->Q[cur].n = 0;
        q->occ[cur >> 6] &= ~(1ull << (cur & 63)); if (!q->occ[cur >> 6]) q->occ1 &= ~(1ull << (cur >> 6));
        if (!q->occ1) { q->cur = 0; return 0; }
        const int w = 63 - __builtin_clzll(q->occ1); q->cur = (w << 6) | (63 - __builtin_clzll(q->occ[w]));
    }
    if (q->cur <= 0) return 0;
    uq_lev *l = &q->Q[q->cur]; *i = l->a[l->h++]; *ahead = l->h + UQ_PD < l->n ? l->a[l->h + UQ_PD] : UQ_NONE;
    return 1;
}
/* the flood's bounds (field voxels, [0 1)), its lowest confidence level, and (comp) the label it writes on the voxels it reaches */
typedef struct { int z0, z1, y0, y1, x0, x1, minb; uint32_t *comp, cid; } uw_box;
/* pop until the queue is empty, each popped voxel giving its unreached neighbours (in the box, level >= minb) the integer within half a
   turn of its own; returns the voxels newly reached */
static size_t uw_flood(uw_ctx *x, uw_queue *q, const uw_box *bx) {
    uw_vox *V = x->V; const uw_geo *g = &x->g; size_t reached = 0, over = 0; uint32_t i, pf;
    while (uq_pop(q, &i, &pf)) {
        if (pf != UQ_NONE) {   /* a voxel UQ_PD pops ahead: its record and its z / y neighbours' (x ones share its line or the next) */
            int pz_, py_, px_; ub_zyx(g, pf, &pz_, &py_, &px_);
            __builtin_prefetch(&V[pf]);
            if (pz_ > 0) __builtin_prefetch(&V[ub_id(g, pz_ - 1, py_, px_)]);
            if (pz_ + 1 < g->D) __builtin_prefetch(&V[ub_id(g, pz_ + 1, py_, px_)]);
            if (py_ > 0) __builtin_prefetch(&V[ub_id(g, pz_, py_ - 1, px_)]);
            if (py_ + 1 < g->H) __builtin_prefetch(&V[ub_id(g, pz_, py_ + 1, px_)]);
        }
        int z, y, xx; ub_zyx(g, i, &z, &y, &xx);
        const int ok[6] = {z > bx->z0, z + 1 < bx->z1, y > bx->y0, y + 1 < bx->y1, xx > bx->x0, xx + 1 < bx->x1};
        const int nb3[6][3] = {{z - 1, y, xx}, {z + 1, y, xx}, {z, y - 1, xx}, {z, y + 1, xx}, {z, y, xx - 1}, {z, y, xx + 1}};
        const double ui = (double)V[i].uw + (x->kb + V[i].k);
        for (int q6 = 0; q6 < 6; q6++) {
            if (!ok[q6]) continue;
            const uint32_t j = ub_id(g, nb3[q6][0], nb3[q6][1], nb3[q6][2]); uw_vox *vj = &V[j];
            if (!vj->bk || vj->bk < bx->minb || vj->k != UK_UNSET) continue;
            const int32_t kr = (int32_t)round(ui - vj->uw) - x->kb;
            if (kr <= UK_UNSET || kr > INT16_MAX) { over++; continue; }
            vj->k = (int16_t)kr; reached++;
            if (bx->comp) bx->comp[j] = bx->cid;
            uq_push(q, j, vj->bk);
        }
    }
    if (over) atomic_fetch_add(&x->over, over);
    return reached;
}
/* the integer (minus the base) that puts the voxel's U nearest the model's; INT32_MIN: out of the 16-bit range */
static int32_t uw_model_k(const uw_ctx *x, int z, int y, int xx, float uw) {
    const fit_fields *f = x->f; const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
    const double th = atan2(py - axis_lerp(f->cy, x->na, pz), px - axis_lerp(f->cx, x->na, pz));
    const int32_t kr = (int32_t)round(fit_W(x->m, pz, py, px) + x->m->dir * th / (2 * M_PI) - uw) - x->kb;
    return kr > UK_UNSET && kr <= INT16_MAX ? kr : INT32_MIN;
}

/* Two-level unwrapping (--uw-split S): the serial flood is bound by memory latency and needs every record in one address space (the
   whole scroll at 19.2 um is ~100x this slab). (A) Tiles of uw_tile^3 field voxels in parallel: each floods its voxels of confidence
   >= S into connected pieces, each seeded at its most confident voxel with the integer nearest the model. (B) Pieces in neighbouring
   tiles are joined by votes over their touching voxel pairs (offset round(U_i - U_j), weight the lower confidence level; a pair of
   pieces scores its best offset's weight minus the runner-up's), strongest first, into a union-find with integer potentials (a
   maximum spanning forest; a vote against a joined pair is ignored). The largest tree keeps its integers; the other pieces are reset.
   (C) The serial flood from that tree's border reaches the rest (low confidence, the reset pieces) in confidence order. A tile commits
   only confident voxels, so no tile is forced through its low-confidence patches (the earlier tiled floods flooded everything per tile
   and lost 4-6 points of winding ids). Measured on the 3.9 cm slab with prod19v2 fields (serial: flood 51 s, 38.5% ids): split 0.7
   39.4% but the main tree holds only 72M of the 323M voxels, so the serial pass still takes 45 s; split 0.5 floods in 27 s but gets
   29.5% (pieces commit 0.5-0.7 voxels whose integers depend on the path). Off by default. */
typedef struct { uint32_t a, b; int32_t d; uint32_t w; } uw_edge;
typedef struct {
    uw_ctx *x; int T, ntz, nty, ntx, thrb; uint32_t *comp; _Atomic int next;
    uint32_t *ncomp, **csize;               /* per tile: its pieces (labels 1..n) and their voxel counts */
    uint32_t *tbase;                        /* per tile: its pieces' first global id */
    uw_edge *e[64]; size_t ne[64], ce[64];  /* per thread: border votes */
    int32_t *shift;                         /* per global piece: its integer shift (INT32_MIN: reset) */
    uint32_t **bnd; size_t *nb;             /* phase C: per z range, the reached voxels next to unreached ones */
} uw2_ctx;
static void uw2_tile_box(const uw2_ctx *u, int t, uw_box *b) {
    const uw_geo *g = &u->x->g; const int tx = t % u->ntx, ty = (t / u->ntx) % u->nty, tz = t / (u->ntx * u->nty);
    b->z0 = tz * u->T; b->z1 = b->z0 + u->T < g->D ? b->z0 + u->T : g->D;
    b->y0 = ty * u->T; b->y1 = b->y0 + u->T < g->H ? b->y0 + u->T : g->H;
    b->x0 = tx * u->T; b->x1 = b->x0 + u->T < g->W ? b->x0 + u->T : g->W;
}
static void uw2_pieces(void *a, int z0_, int z1_, int th) {   /* (A): tiles off a shared counter */
    uw2_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; uw_vox *V = x->V; (void)z0_; (void)z1_; (void)th;
    const int nt = u->ntz * u->nty * u->ntx; uw_queue q; if (uq_init(&q)) abort();
    uint32_t *ids = nullptr; size_t cap = 0;
    for (int t; (t = atomic_fetch_add(&u->next, 1)) < nt;) {
        uw_box b; uw2_tile_box(u, t, &b);
        const size_t nv = (size_t)(b.z1 - b.z0) * (b.y1 - b.y0) * (b.x1 - b.x0);
        if (nv > cap) { cap = nv; free(ids); ids = malloc(cap * sizeof *ids); if (!ids) abort(); }
        size_t cnt[UQ_LEVELS] = {0}, n = 0;   /* the confident voxels, most confident first (a counting sort by level) */
        for (int z = b.z0; z < b.z1; z++) for (int y = b.y0; y < b.y1; y++) for (int xx = b.x0; xx < b.x1; xx++) {
            const int bk = V[ub_id(g, z, y, xx)].bk; if (bk >= u->thrb) cnt[bk]++;
        }
        for (int l = UQ_LEVELS - 1; l >= 0; l--) { const size_t c = cnt[l]; cnt[l] = n; n += c; }
        for (int z = b.z0; z < b.z1; z++) for (int y = b.y0; y < b.y1; y++) for (int xx = b.x0; xx < b.x1; xx++) {
            const uint32_t id = ub_id(g, z, y, xx); const int bk = V[id].bk; if (bk >= u->thrb) ids[cnt[bk]++] = id;
        }
        uint32_t np = 0, capp = 0, *sz = nullptr;
        b.minb = u->thrb; b.comp = u->comp;
        for (size_t s = 0; s < n; s++) {
            const uint32_t id = ids[s]; if (V[id].k != UK_UNSET) continue;
            int z, y, xx; ub_zyx(g, id, &z, &y, &xx);
            const int32_t k = uw_model_k(x, z, y, xx, V[id].uw); if (k == INT32_MIN) { atomic_fetch_add(&x->over, 1); continue; }
            if (np == capp) { capp = capp ? 2 * capp : 256; sz = realloc(sz, capp * sizeof *sz); if (!sz) abort(); }
            V[id].k = (int16_t)k; u->comp[id] = b.cid = ++np; uq_push(&q, id, V[id].bk);
            sz[np - 1] = 1 + (uint32_t)uw_flood(x, &q, &b);
        }
        u->ncomp[t] = np; u->csize[t] = sz;
    }
    free(ids); uq_free(&q);
}
static void uw2_votes(void *a, int t0, int t1, int th) {   /* (B): tiles [t0, t1), their +z / +y / +x faces */
    uw2_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; const uw_vox *V = x->V;
    for (int t = t0; t < t1; t++) {
        uw_box b; uw2_tile_box(u, t, &b);
        const int tx = t % u->ntx, ty = (t / u->ntx) % u->nty, tz = t / (u->ntx * u->nty);
        for (int ax = 0; ax < 3; ax++) {
            const int last = ax == 0 ? tz + 1 >= u->ntz : ax == 1 ? ty + 1 >= u->nty : tx + 1 >= u->ntx; if (last) continue;
            const int t2 = ax == 0 ? t + u->ntx * u->nty : ax == 1 ? t + u->ntx : t + 1;
            int lo[3] = {b.z0, b.y0, b.x0}, hi[3] = {b.z1, b.y1, b.x1}; lo[ax] = hi[ax] - 1;   /* the tile's last layer along ax */
            for (int z = lo[0]; z < hi[0]; z++) for (int y = lo[1]; y < hi[1]; y++) for (int xx = lo[2]; xx < hi[2]; xx++) {
                const uint32_t i = ub_id(g, z, y, xx); if (!u->comp[i]) continue;
                const uint32_t j = ub_id(g, z + (ax == 0), y + (ax == 1), xx + (ax == 2)); if (!u->comp[j]) continue;
                uint32_t pa = u->tbase[t] + u->comp[i] - 1, pb = u->tbase[t2] + u->comp[j] - 1;
                int32_t d = (int32_t)round(((double)V[i].uw + V[i].k) - ((double)V[j].uw + V[j].k));   /* pot_b - pot_a */
                if (pa > pb) { const uint32_t s = pa; pa = pb; pb = s; d = -d; }
                if (u->ne[th] == u->ce[th]) { u->ce[th] = u->ce[th] ? 2 * u->ce[th] : 65536; u->e[th] = realloc(u->e[th], u->ce[th] * sizeof *u->e[th]); if (!u->e[th]) abort(); }
                u->e[th][u->ne[th]++] = (uw_edge){pa, pb, d, (uint32_t)(V[i].bk < V[j].bk ? V[i].bk : V[j].bk)};
            }
        }
    }
}
static void uw2_apply(void *a, int t0, int t1, int th) {   /* (B'): the joined pieces shifted, the rest reset */
    uw2_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; uw_vox *V = x->V; (void)th;
    for (int t = t0; t < t1; t++) {
        uw_box b; uw2_tile_box(u, t, &b);
        for (int z = b.z0; z < b.z1; z++) for (int y = b.y0; y < b.y1; y++) for (int xx = b.x0; xx < b.x1; xx++) {
            const uint32_t i = ub_id(g, z, y, xx); if (!u->comp[i]) continue;
            const int32_t s = u->shift[u->tbase[t] + u->comp[i] - 1], k = s == INT32_MIN ? INT32_MIN : V[i].k + s;
            if (k == INT32_MIN || k <= UK_UNSET || k > INT16_MAX) V[i].k = UK_UNSET; else V[i].k = (int16_t)k;
        }
    }
}
static void uw2_border(void *a, int z0, int z1, int th) {   /* (C): reached voxels with an unreached floodable neighbour, by z range */
    uw2_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; const uw_vox *V = x->V; size_t n = 0, cap = 0; uint32_t *o = nullptr;
    for (int z = z0; z < z1; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) {
        const uint32_t i = ub_id(g, z, y, xx); if (V[i].k == UK_UNSET) continue;
        const int ok[6] = {z > 0, z + 1 < g->D, y > 0, y + 1 < g->H, xx > 0, xx + 1 < g->W};
        const int nb3[6][3] = {{z - 1, y, xx}, {z + 1, y, xx}, {z, y - 1, xx}, {z, y + 1, xx}, {z, y, xx - 1}, {z, y, xx + 1}};
        int open = 0;
        for (int q6 = 0; q6 < 6 && !open; q6++) if (ok[q6]) { const uw_vox *v = &V[ub_id(g, nb3[q6][0], nb3[q6][1], nb3[q6][2])]; open = v->bk && v->k == UK_UNSET; }
        if (!open) continue;
        if (n == cap) { cap = cap ? 2 * cap : 65536; o = realloc(o, cap * sizeof *o); if (!o) abort(); }
        o[n++] = i;
    }
    u->bnd[th] = o; u->nb[th] = n;
}
static int uw_edge_cmp(const void *p, const void *q) {
    const uw_edge *a = p, *b = q;
    return a->a != b->a ? (a->a < b->a ? -1 : 1) : a->b != b->b ? (a->b < b->b ? -1 : 1) : a->d != b->d ? (a->d < b->d ? -1 : 1) : 0;
}
typedef struct { uint32_t a, b; int32_t d; double s; } uw_pair;
static int uw_pair_cmp(const void *p, const void *q) {   /* strongest first; ties by the pieces */
    const uw_pair *a = p, *b = q;
    return a->s != b->s ? (a->s > b->s ? -1 : 1) : a->a != b->a ? (a->a < b->a ? -1 : 1) : a->b != b->b ? (a->b < b->b ? -1 : 1) : 0;
}
static uint32_t uf_find(uint32_t *par, int32_t *pot, uint32_t a, int32_t *pa) {   /* root; *pa: a's potential relative to it */
    uint32_t r = a; int32_t s = 0;
    while (par[r] != r) { s += pot[r]; r = par[r]; }
    const int32_t sa = s;
    for (uint32_t c = a; par[c] != c;) { const uint32_t nx = par[c]; const int32_t p = pot[c]; par[c] = r; pot[c] = s; s -= p; c = nx; }
    *pa = sa; return r;
}
/* (A) - (C) on the records set up by fit_unwrap_voxels (no seed model); returns the voxels reached */
static size_t uw_two_level(uw_ctx *x, double split, int T, size_t nr, int verbose) {
    const uw_geo *g = &x->g; const double t0 = nn_wall_seconds();
    uw2_ctx u = {.x = x, .T = T, .ntz = (g->D + T - 1) / T, .nty = (g->H + T - 1) / T, .ntx = (g->W + T - 1) / T};
    u.thrb = 1 + (int)(fmin(split, 1.0) * (UQ_LEVELS - 2));
    const int nt = u.ntz * u.nty * u.ntx;
    u.comp = calloc(nr, sizeof *u.comp); u.ncomp = calloc(nt, sizeof *u.ncomp); u.csize = calloc(nt, sizeof *u.csize); u.tbase = calloc(nt + 1, sizeof *u.tbase);
    if (!u.comp || !u.ncomp || !u.csize || !u.tbase) { fprintf(stderr, "fit: out of memory for the two-level unwrapping\n"); abort(); }
    fit_zpar(x->nt, x->nt, uw2_pieces, &u);   /* one call per thread; tiles off u.next */
    uint64_t np = 0; for (int t = 0; t < nt; t++) { u.tbase[t] = (uint32_t)np; np += u.ncomp[t]; }
    if (np >= UINT32_MAX) { fprintf(stderr, "fit: too many unwrapping pieces\n"); abort(); }
    uint32_t *par = malloc(np * sizeof *par); int32_t *pot = calloc(np, sizeof *pot); uint64_t *sz = malloc(np * sizeof *sz);
    for (int t = 0; t < nt; t++) for (uint32_t k = 0; k < u.ncomp[t]; k++) { par[u.tbase[t] + k] = u.tbase[t] + k; sz[u.tbase[t] + k] = u.csize[t][k]; }
    const double t1 = nn_wall_seconds();
    fit_zpar(nt, x->nt, uw2_votes, &u);
    size_t ne = 0; for (int t = 0; t < x->nt; t++) ne += u.ne[t];
    uw_edge *E = malloc((ne ? ne : 1) * sizeof *E); ne = 0;
    for (int t = 0; t < x->nt; t++) { if (u.ne[t]) memcpy(E + ne, u.e[t], u.ne[t] * sizeof *E); ne += u.ne[t]; free(u.e[t]); }
    qsort(E, ne, sizeof *E, uw_edge_cmp);
    size_t npair = 0; uw_pair *P = malloc((ne ? ne : 1) * sizeof *P);
    for (size_t s = 0; s < ne;) {   /* per piece pair: the offsets' summed weights; best minus runner-up */
        size_t e = s; double best = 0, second = 0; int32_t bd = 0;
        while (e < ne && E[e].a == E[s].a && E[e].b == E[s].b) {
            size_t f2 = e; double w = 0; while (f2 < ne && E[f2].a == E[e].a && E[f2].b == E[e].b && E[f2].d == E[e].d) w += E[f2++].w;
            if (w > best) { second = best; best = w; bd = E[e].d; } else if (w > second) second = w;
            e = f2;
        }
        P[npair++] = (uw_pair){E[s].a, E[s].b, bd, best - second}; s = e;
    }
    free(E); qsort(P, npair, sizeof *P, uw_pair_cmp);
    size_t joined = 0;
    for (size_t k = 0; k < npair; k++) {
        if (P[k].s <= 0) break;
        int32_t pa, pb; const uint32_t ra = uf_find(par, pot, P[k].a, &pa), rb = uf_find(par, pot, P[k].b, &pb);
        if (ra == rb) continue;
        if (sz[ra] >= sz[rb]) { par[rb] = ra; pot[rb] = pa + P[k].d - pb; sz[ra] += sz[rb]; }   /* pot_b(total) = pot_a + d */
        else { par[ra] = rb; pot[ra] = pb - P[k].d - pa; sz[rb] += sz[ra]; }
        joined++;
    }
    free(P);
    uint32_t main = UINT32_MAX; uint64_t msz = 0;
    for (uint32_t k = 0; k < np; k++) if (par[k] == k && sz[k] > msz) { msz = sz[k]; main = k; }
    u.shift = malloc((np ? np : 1) * sizeof *u.shift);
    for (uint32_t k = 0; k < np; k++) { int32_t p; u.shift[k] = uf_find(par, pot, k, &p) == main ? p : INT32_MIN; }
    fit_zpar(nt, x->nt, uw2_apply, &u);
    const double t2 = nn_wall_seconds();
    u.bnd = calloc(x->nt, sizeof *u.bnd); u.nb = calloc(x->nt, sizeof *u.nb);
    fit_zpar(g->D, x->nt, uw2_border, &u);
    uw_queue q; if (uq_init(&q)) abort();
    size_t nbd = 0;
    for (int t = 0; t < x->nt; t++) { for (size_t k = 0; k < u.nb[t]; k++) uq_push(&q, u.bnd[t][k], x->V[u.bnd[t][k]].bk); nbd += u.nb[t]; free(u.bnd[t]); }
    const size_t serial = uw_flood(x, &q, &(uw_box){0, g->D, 0, g->H, 0, g->W, 1, nullptr, 0});
    uq_free(&q);
    size_t reached = 0;   /* (the records are few enough to count serially next to the flood) */
    for (size_t i = 0; i < nr; i++) reached += x->V[i].k != UK_UNSET;
    if (verbose) fprintf(stderr, "fit: two-level unwrapping (split %.2f, tiles %d^3 x %d): %llu pieces, %zu joined, main %llu voxels; serial "
                         "%zu more from %zu border voxels; pieces %.1fs, votes %.1fs, serial %.1fs\n", split, T, nt, (unsigned long long)np,
                         joined, (unsigned long long)msz, serial, nbd, t1 - t0, t2 - t1, nn_wall_seconds() - t2);
    for (int t = 0; t < nt; t++) free(u.csize[t]);
    free(u.comp); free(u.ncomp); free(u.csize); free(u.tbase); free(par); free(pot); free(sz); free(u.shift); free(u.bnd); free(u.nb);
    return reached;
}

/* Local slip repair (fit_cfg uw_repair, fitter Phase 4). The quality-ordered flood is a maximum spanning tree: where the phase has a
   dislocation (a loop whose wrapped differences do not sum to zero, e.g. two sheets merged over a patch) the tree closes the loop
   wherever its order happens to, and a whole region beyond can come out a turn off (a "slip": on the held-out Paris 4 box half the
   box in some predict windows, ring-shaped regions 6-10 mm from the axis). Here the flooded voxels of each tile (uw_repair^3 field
   voxels) split into pieces whose integers agree (connected through neighbours within half a turn); every touching voxel pair of two
   pieces votes for their integer offset with the lower confidence as weight, optional evidence adds longer-range votes (the winding
   mod 4 planes; the gradient stages' W, i.e. the normal / density integration; the CT sheet spacing), and the pieces' shifts minimise
   the summed weighted L1 disagreement (src/l1pot.h, exact). A dislocation is then cut along its cheapest (least confident, least
   evidenced) surface. The shifted integers seed the phase stage's delta like the flood's. */
typedef struct {
    uw_ctx *x; const fit_cfg *c; int T, o, ntz, nty, ntx; _Atomic int next;   /* o: the tile grid offset (field voxels, all axes) */
    uint32_t *comp;                         /* per record: its piece within its tile (1..), 0 not flooded */
    uint32_t *npc, *tbase;                  /* per tile: pieces, the first global id */
    l1_vote *v[64]; size_t nv[64], cv[64];  /* per thread: merged votes */
    int pass;                               /* 0 pieces, 1 votes */
    const int32_t *shift;                   /* apply / seams: per global piece */
    uint8_t *seam; int seam_shift;          /* per field voxel: disagreeing neighbours (seam_shift 0 / 4: the low / high nibble) */
    size_t nseam[64];
    int gd; double wgeo, ww4, wcs; int w4c; /* the evidence: distances (field voxels) and weights */
    double qpow, sst;                       /* vote weight: confidence power, stretch scale (0 off) */
    uint8_t *bad;                           /* per record (sst > 0): the stretch weight < 1/2 (pieces do not join such voxels to others) */
    float *lst; int sr;                     /* per record (sst > 0): log2 of the phase stretch; its box radius */
    uint32_t anchor; double wref;           /* fit_cfg uw_rep_ref: the ref anchor node (the piece count) and the vote weight (0 off) */
} uwr_ctx;
static inline uint32_t uwr_gid(const uwr_ctx *u, int z, int y, int xx, uint32_t id) {
    const int t = (((z + u->o) / u->T) * u->nty + (y + u->o) / u->T) * u->ntx + (xx + u->o) / u->T; return u->tbase[t] + u->comp[id] - 1;
}
static inline double uw_U(const uw_ctx *x, uint32_t id) { return (double)x->V[id].uw + x->V[id].k; }   /* (the base kb cancels in differences) */
static void uwr_vote(uwr_ctx *u, int th, uint32_t a, uint32_t b, int32_t d, int64_t w) {
    if (a == b || w <= 0) return;
    if (u->nv[th] == u->cv[th]) { u->cv[th] = u->cv[th] ? 2 * u->cv[th] : 65536; u->v[th] = realloc(u->v[th], u->cv[th] * sizeof *u->v[th]); if (!u->v[th]) abort(); }
    u->v[th][u->nv[th]++] = (l1_vote){a, b, d, w};
}
/* the model's U - dir theta / 2 pi = W and theta at a field voxel */
static double uwr_Wm(const uw_ctx *x, int z, int y, int xx, double *th) {
    const fit_fields *f = x->f; const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
    *th = atan2(py - axis_lerp(f->cy, x->na, pz), px - axis_lerp(f->cx, x->na, pz));
    return fit_W(x->m, pz, py, px);
}
/* the phase stretch at a field voxel: |grad phase| (turns per field voxel) x the spacing (field voxels per turn), both over the (2r+1)^3
   box around it (the gradient per axis from the summed phasor products z(+e) conj(z(-e)), i.e. confidence-weighted; the mean spacing):
   ~1 where the phase counts every sheet, < 1 where it skipped one, > 1 where it added one; 1 when unknown. r 1 flags 4% of the cells
   whose phase agrees with the published wraps on the Paris 4 hold box against 8% at r 0 (and 36% vs 41% of those > 1/4 turn off). */
static double uwr_stretch(const fit_fields *f, int z, int y, int xx, int r) {
    const size_t HW = (size_t)f->H * f->W; const size_t st[3] = {HW, (size_t)f->W, 1}; const int dim[3] = {f->D, f->H, f->W};
    double pr[3] = {0}, pi[3] = {0}, ssp = 0; int nsp = 0;
    for (int a = z - r; a <= z + r; a++) for (int b = y - r; b <= y + r; b++) for (int c = xx - r; c <= xx + r; c++) {
        if (a < 0 || b < 0 || c < 0 || a >= f->D || b >= f->H || c >= f->W) continue;
        const size_t i = (size_t)a * HW + (size_t)b * f->W + c; const int pos[3] = {a, b, c};
        if (f->sp[i]) { ssp += f->sp[i]; nsp++; }
        for (int d = 0; d < 3; d++) {
            if (pos[d] <= 0 || pos[d] >= dim[d] - 1) continue;
            const size_t im = i - st[d], ip = i + st[d];
            const double cm = f->pc[im] - 128, sm = f->ps[im] - 128, cp = f->pc[ip] - 128, sq = f->ps[ip] - 128;
            pr[d] += cp * cm + sq * sm; pi[d] += sq * cm - cp * sm;   /* z(+) conj z(-) */
        }
    }
    if (!nsp) return 1;
    double g2 = 0; int n = 0;
    for (int d = 0; d < 3; d++) if (pr[d] != 0 || pi[d] != 0) { const double e = atan2(pi[d], pr[d]) / (4 * M_PI); g2 += e * e; n++; }
    if (!n) return 1;
    return sqrt(g2 * 3.0 / n) * ssp / nsp / 8.0;   /* (missing axes: the mean square of the others) */
}
static double wrap_pi(double a) { return a - 2 * M_PI * floor((a + M_PI) / (2 * M_PI)); }
static int64_t uwr_w(const uwr_ctx *u, int bk, int z, int y, int xx, int zn, int yn, int xn) {   /* a phase vote's weight */
    double w = u->qpow == 1 ? bk : (UQ_LEVELS - 2) * pow(bk / (double)(UQ_LEVELS - 2), u->qpow);
    if (u->sst > 0) {
        const uw_geo *g = &u->x->g; const float a = u->lst[ub_id(g, z, y, xx)], b = u->lst[ub_id(g, zn, yn, xn)];
        w *= exp(-(a * a + b * b) / (2 * u->sst * u->sst));
    }
    return w < 1 ? 1 : (int64_t)lround(w);
}
/* embedding evidence (fit_cfg emb): from voxel i, partners uw_rep_emb_d native voxels away along the model's two sheet tangents; where
   the partner is in another piece, the embedding's integer for the same sheet (src/fit_emb.h emb_shift_vote: the sheet at i's level L
   is the one j calls L + r, so W(j) - W(i) should be r turns lower than flooded) votes -r between the pieces */
static int uwr_side(const uw_ctx *x, int z, int y, int xx, double Wf, emb_side *s) {
    const fit_fields *f = x->f; const double h = f->h, pz = (z + 0.5) * h, py = (y + 0.5) * h, px = (xx + 0.5) * h, e = h;
    double g[3] = {fit_W(x->m, pz + e, py, px) - fit_W(x->m, pz - e, py, px), fit_W(x->m, pz, py + e, px) - fit_W(x->m, pz, py - e, px),
                   fit_W(x->m, pz, py, px + e) - fit_W(x->m, pz, py, px - e)};
    const double gl = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); if (!(gl > 1e-9)) return -1;
    for (int d = 0; d < 3; d++) s->n[d] = g[d] / gl;
    s->sp = 2 * e / gl;   /* native voxels per turn */
    if (!(s->sp > 1 && s->sp < 1e3)) return -1;
    s->p[0] = pz + x->c->emb_zoff; s->p[1] = py; s->p[2] = px; s->W = Wf;
    return 0;
}
static void uwr_emb(uwr_ctx *u, int th, int z, int y, int xx, uint32_t i, uint32_t ga, double ui) {
    uw_ctx *x = u->x; const uw_geo *g = &x->g; const fit_cfg *c = u->c; const fit_fields *f = x->f; const uw_vox *V = x->V;
    emb_side a; double th0; uwr_Wm(x, z, y, xx, &th0);
    if (uwr_side(x, z, y, xx, 0.0, &a)) return;   /* W(i) = 0: only differences matter */
    double t[2][3];
    {   const double ez[3] = {1, 0, 0}, ey[3] = {0, 1, 0}; const double *r0 = fabs(a.n[0]) < 0.9 ? ez : ey;
        t[0][0] = a.n[1] * r0[2] - a.n[2] * r0[1]; t[0][1] = a.n[2] * r0[0] - a.n[0] * r0[2]; t[0][2] = a.n[0] * r0[1] - a.n[1] * r0[0];
        const double l = sqrt(t[0][0] * t[0][0] + t[0][1] * t[0][1] + t[0][2] * t[0][2]); for (int d = 0; d < 3; d++) t[0][d] /= l;
        t[1][0] = a.n[1] * t[0][2] - a.n[2] * t[0][1]; t[1][1] = a.n[2] * t[0][0] - a.n[0] * t[0][2]; t[1][2] = a.n[0] * t[0][1] - a.n[1] * t[0][0]; }
    for (int k = 0; k < c->uw_rep_emb_nd; k++) for (int dd = 0; dd < 2; dd++) {
        const double D = c->uw_rep_emb_d[k];
        const int zn = (int)floor((a.p[0] + D * t[dd][0]) / f->h), yn = (int)floor((a.p[1] + D * t[dd][1]) / f->h), xn = (int)floor((a.p[2] + D * t[dd][2]) / f->h);
        if (zn < 0 || yn < 0 || xn < 0 || zn >= g->D || yn >= g->H || xn >= g->W) continue;
        const uint32_t j = ub_id(g, zn, yn, xn); if (V[j].k == UK_UNSET) continue;
        const uint32_t gb = uwr_gid(u, zn, yn, xn, j); if (ga == gb) continue;
        double th1; uwr_Wm(x, zn, yn, xn, &th1);
        const double dWc = (uw_U(x, j) - ui) - x->m->dir * wrap_pi(th1 - th0) / (2 * M_PI);   /* the flooded W(j) - W(i) */
        emb_side b; if (uwr_side(x, zn, yn, xn, dWc, &b)) continue;
        int r = 0; float cf = 0;
        if (emb_shift_vote(c->emb, &a, &b, c->uw_rep_emb_rmax, c->emb_phi, &r, &cf)) continue;
        const int bq = V[i].bk < V[j].bk ? V[i].bk : V[j].bk;
        const double wv = c->uw_rep_emb * bq * fmin(1.0, cf / c->uw_rep_emb_m0);
        if (wv >= 1) uwr_vote(u, th, ga, gb, (int32_t)-r, (int64_t)lround(wv));
    }
}
static void uwr_tiles(void *a, int z0_, int z1_, int th) {
    uwr_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; const uw_vox *V = x->V; (void)z0_; (void)z1_;
    const int nt = u->ntz * u->nty * u->ntx; uint32_t *q = nullptr; size_t qcap = 0;
    for (int t; (t = atomic_fetch_add(&u->next, 1)) < nt;) {
        const int tx = t % u->ntx, ty = (t / u->ntx) % u->nty, tz = t / (u->ntx * u->nty);
        const int za = tz * u->T - u->o, ya = ty * u->T - u->o, xa = tx * u->T - u->o;
        const int z0 = za > 0 ? za : 0, y0 = ya > 0 ? ya : 0, x0 = xa > 0 ? xa : 0;
        const int z1 = za + u->T < g->D ? za + u->T : g->D, y1 = ya + u->T < g->H ? ya + u->T : g->H, x1 = xa + u->T < g->W ? xa + u->T : g->W;
        if (u->pass == 0) {   /* pieces: flooded voxels connected through neighbours whose integers agree */
            const size_t nv = (size_t)(z1 - z0) * (y1 - y0) * (x1 - x0);
            if (nv > qcap) { qcap = nv; free(q); q = malloc(qcap * sizeof *q); if (!q) abort(); }
            uint32_t np = 0;
            for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) for (int xx = x0; xx < x1; xx++) {
                const uint32_t id = ub_id(g, z, y, xx); if (V[id].k == UK_UNSET || u->comp[id]) continue;
                u->comp[id] = ++np; size_t qh = 0, qt = 0; q[qt++] = id;
                while (qh < qt) {
                    const uint32_t i = q[qh++]; int zz, yy, xq; ub_zyx(g, i, &zz, &yy, &xq); const double ui = uw_U(x, i);
                    const int ok[6] = {zz > z0, zz + 1 < z1, yy > y0, yy + 1 < y1, xq > x0, xq + 1 < x1};
                    const int nb3[6][3] = {{zz - 1, yy, xq}, {zz + 1, yy, xq}, {zz, yy - 1, xq}, {zz, yy + 1, xq}, {zz, yy, xq - 1}, {zz, yy, xq + 1}};
                    for (int k = 0; k < 6; k++) {
                        if (!ok[k]) continue;
                        const uint32_t j = ub_id(g, nb3[k][0], nb3[k][1], nb3[k][2]);
                        if (V[j].k == UK_UNSET || u->comp[j] || fabs(ui - uw_U(x, j)) >= 0.5 || (u->bad && u->bad[i] != u->bad[j])) continue;
                        u->comp[j] = np; q[qt++] = j;
                    }
                }
            }
            u->npc[t] = np;
        } else if (u->pass == 1) {   /* votes: touching pairs (+z +y +x neighbours) of two pieces, then the evidence */
            const size_t n0 = u->nv[th];
            for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) for (int xx = x0; xx < x1; xx++) {
                const uint32_t i = ub_id(g, z, y, xx); if (V[i].k == UK_UNSET) continue;
                const uint32_t ga = uwr_gid(u, z, y, xx, i); const double ui = uw_U(x, i);
                for (int ax = 0; ax < 3; ax++) {
                    const int zn = z + (ax == 0), yn = y + (ax == 1), xn = xx + (ax == 2);
                    if (zn >= g->D || yn >= g->H || xn >= g->W) continue;
                    const uint32_t j = ub_id(g, zn, yn, xn); if (V[j].k == UK_UNSET) continue;
                    const uint32_t gb = uwr_gid(u, zn, yn, xn, j); if (ga == gb) continue;
                    uwr_vote(u, th, ga, gb, (int32_t)lround(ui - uw_U(x, j)), uwr_w(u, V[i].bk < V[j].bk ? V[i].bk : V[j].bk, z, y, xx, zn, yn, xn));
                }
                if (u->c->emb && u->c->uw_rep_emb > 0 && z % u->c->uw_rep_emb_sub == 0 && y % u->c->uw_rep_emb_sub == 0 && xx % u->c->uw_rep_emb_sub == 0) uwr_emb(u, th, z, y, xx, i, ga, ui);
                /* evidence: votes between voxels farther apart (every other voxel per axis) */
                if ((z | y | xx) & 1) continue;
                if (u->wref > 0) {   /* the ref anchor: this voxel's integer offset to the ref (s_anchor - s_piece = -offset) */
                    const double r = uw_Uref(x, z, y, xx) - x->rbias - (ui + x->kb); const int64_t wv = llround(u->wref * V[i].bk);
                    if (wv >= 1) uwr_vote(u, th, ga, u->anchor, -(int32_t)lround(r), wv);
                }
                const fit_fields *f = x->f; const size_t fi = ((size_t)z * f->H + y) * f->W + xx;
                for (int ev = 0; ev < 3; ev++) {
                    const double wt = ev == 0 ? u->ww4 : ev == 1 ? u->wgeo : u->wcs; if (wt <= 0) continue;
                    if (ev == 0 && !f->wd[3][fi]) continue;
                    if (ev == 2 && !f->cs[fi]) continue;
                    const int D = ev == 0 ? u->w4c : u->gd; if (D < 1) continue;
                    for (int ax = 0; ax < 3; ax++) {
                        const int zn = z + (ax == 0) * D, yn = y + (ax == 1) * D, xn = xx + (ax == 2) * D;
                        if (zn >= g->D || yn >= g->H || xn >= g->W) continue;
                        const uint32_t j = ub_id(g, zn, yn, xn); if (V[j].k == UK_UNSET) continue;
                        const uint32_t gb = uwr_gid(u, zn, yn, xn, j); if (ga == gb) continue;
                        double th0, th1; const double Wm0 = uwr_Wm(x, z, y, xx, &th0), Wm1 = uwr_Wm(x, zn, yn, xn, &th1);
                        const double dth = wrap_pi(th1 - th0) / (2 * M_PI), dWc = (uw_U(x, j) - ui) - x->m->dir * dth;   /* the flooded W difference */
                        double want;   /* the evidence's W(j) - W(i) */
                        if (ev == 0) {   /* mod 4: the candidate nearest the model's difference */
                            const double dl = f->wd[ax][fi] / 64.0, dm = Wm1 - Wm0; want = dl + 4 * round((dm - dl) / 4);
                        } else if (ev == 1) {   /* uw_rep_geo_init: the --init-model's W (a global coarse fit) instead of this fit's gradient stages */
                            if (u->c->uw_rep_geo_init && u->c->init) {
                                const fit_model *gm = u->c->init; const double *o = u->c->init_off, h = f->h;
                                want = fit_W(gm, (zn + 0.5) * h + o[0], (yn + 0.5) * h + o[1], (xn + 0.5) * h + o[2]) - fit_W(gm, (z + 0.5) * h + o[0], (y + 0.5) * h + o[1], (xx + 0.5) * h + o[2]);
                            } else want = Wm1 - Wm0;
                        }
                        else {   /* turns along the model's gradient from the CT spacing (mean of the two ends' spacings) */
                            const size_t fj = ((size_t)zn * f->H + yn) * f->W + xn; if (!f->cs[fj]) continue;
                            const double spn = 0.5 * (f->cs[fi] + f->cs[fj]) * f->h / 8.0;   /* native voxels per turn */
                            double gv[3]; const double e = f->h, pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
                            gv[0] = fit_W(x->m, pz + e, py, px) - fit_W(x->m, pz - e, py, px); gv[1] = fit_W(x->m, pz, py + e, px) - fit_W(x->m, pz, py - e, px);
                            gv[2] = fit_W(x->m, pz, py, px + e) - fit_W(x->m, pz, py, px - e);
                            const double gl = sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]); if (gl < 1e-9) continue;
                            want = gv[ax] / gl * D * f->h / spn;
                        }
                        const double r = want - dWc, rr = round(r);
                        if (fabs(r - rr) > 0.3) continue;   /* the evidence disagrees with the phase by a fraction: no integer vote */
                        const int bq = V[i].bk < V[j].bk ? V[i].bk : V[j].bk;
                        const double wv = wt * bq * (ev == 0 ? f->wd[3][fi] / 255.0 : 1.0);
                        if (wv >= 1) uwr_vote(u, th, ga, gb, (int32_t)rr, (int64_t)lround(wv));
                    }
                }
            }
            /* merge this tile's votes in place (few distinct pairs per tile) */
            const size_t m = l1_votes_merge(u->v[th] + n0, u->nv[th] - n0); u->nv[th] = n0 + m;
        }
    }
    free(q);
}
static void uwr_badness(void *a, int z0, int z1, int th) {   /* the stretched / compressed voxels (sst > 0) */
    uwr_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g;
    for (int z = z0; z < z1; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) {
        const uint32_t i = ub_id(g, z, y, xx); if (x->V[i].k == UK_UNSET) continue;
        const double l = log2(fmax(uwr_stretch(x->f, z, y, xx, u->sr), 1e-3)); u->lst[i] = (float)l; u->bad[i] = exp(-l * l / (2 * u->sst * u->sst)) < 0.5;
    }
}
static void uwr_apply(void *a, int z0, int z1, int th) {   /* the shifts, and the seams (disagreeing neighbour pairs) */
    uwr_ctx *u = a; uw_ctx *x = u->x; const uw_geo *g = &x->g; uw_vox *V = x->V; size_t ns = 0;
    for (int z = z0; z < z1; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) {
        const uint32_t i = ub_id(g, z, y, xx); if (V[i].k == UK_UNSET) continue;
        if (u->shift) {
            const int32_t k = V[i].k + u->shift[uwr_gid(u, z, y, xx, i)];
            V[i].k = (int16_t)(k <= UK_UNSET ? UK_UNSET + 1 : k > INT16_MAX ? INT16_MAX : k);
        }
    }
    /* the seams (shift null: a separate pass, as the neighbours of other z ranges must be shifted first) */
    if (!u->shift) for (int z = z0; z < z1; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) {
        const uint32_t i = ub_id(g, z, y, xx); if (V[i].k == UK_UNSET) continue;
        const double ui = uw_U(x, i); int cnt = 0;
        const int ok[6] = {z > 0, z + 1 < g->D, y > 0, y + 1 < g->H, xx > 0, xx + 1 < g->W};
        const int nb3[6][3] = {{z - 1, y, xx}, {z + 1, y, xx}, {z, y - 1, xx}, {z, y + 1, xx}, {z, y, xx - 1}, {z, y, xx + 1}};
        for (int k = 0; k < 6; k++) if (ok[k]) { const uint32_t j = ub_id(g, nb3[k][0], nb3[k][1], nb3[k][2]); if (V[j].k != UK_UNSET && fabs(ui - uw_U(x, j)) >= 0.5) cnt++; }
        ns += cnt;
        if (u->seam) u->seam[((size_t)z * g->H + y) * g->W + xx] |= (uint8_t)(cnt << u->seam_shift | (u->bad && u->bad[i] && !u->seam_shift ? 8 : 0));
    }
    u->nseam[th] = ns;
}
static void uw_repair1(uw_ctx *x, size_t nr, const fit_cfg *c, int o_) {
    const uw_geo *g = &x->g; const double t0 = nn_wall_seconds(); const fit_fields *f = x->f;
    const int T = c->uw_repair;
    const int o = o_ > 0 ? o_ % T : 0;
    uwr_ctx u = {.x = x, .c = c, .T = T, .o = o, .ntz = (g->D + o + T - 1) / T, .nty = (g->H + o + T - 1) / T, .ntx = (g->W + o + T - 1) / T};
    const int nt = u.ntz * u.nty * u.ntx;
    u.comp = calloc(nr, sizeof *u.comp); u.npc = calloc(nt, sizeof *u.npc); u.tbase = calloc(nt + 1, sizeof *u.tbase);
    if (!u.comp || !u.npc || !u.tbase) { fprintf(stderr, "fit: out of memory for the slip repair (skipped)\n"); free(u.comp); free(u.npc); free(u.tbase); return; }
    const size_t nf = (size_t)g->D * g->H * g->W;
    if (c->uw_rep_dump) u.seam = calloc(nf, 1);
    u.qpow = c->uw_rep_qpow > 0 ? c->uw_rep_qpow : 1; u.sst = c->uw_rep_stretch;
    u.sr = c->uw_rep_sr;
    if (u.sst > 0) {
        u.bad = calloc(nr, 1); u.lst = calloc(nr, sizeof *u.lst);
        if (!u.bad || !u.lst) { free(u.bad); free(u.lst); u.bad = nullptr; u.lst = nullptr; u.sst = 0; fprintf(stderr, "fit: no memory for the repair's stretch (off)\n"); }
        else fit_zpar(g->D, x->nt, uwr_badness, &u);
    }
    fit_zpar(g->D, x->nt, uwr_apply, &u);   /* the flood's seams (shift null: count only) */
    size_t seam0 = 0; for (int t = 0; t < x->nt; t++) seam0 += u.nseam[t];
    u.pass = 0; fit_zpar(x->nt, x->nt, uwr_tiles, &u);
    uint64_t np = 0; for (int t = 0; t < nt; t++) { u.tbase[t] = (uint32_t)np; np += u.npc[t]; }
    if (np >= UINT32_MAX) { fprintf(stderr, "fit: too many repair pieces (skipped)\n"); free(u.comp); free(u.npc); free(u.tbase); free(u.seam); free(u.bad); free(u.lst); return; }
    u.gd = c->uw_rep_gd > 0 ? c->uw_rep_gd : 8; u.wgeo = c->uw_rep_geo; u.wcs = f->cs ? c->uw_rep_cs : 0;
    u.ww4 = f->wd[0] && f->wd[1] && f->wd[2] && f->wd[3] && f->w4_d > 0 ? c->uw_rep_w4 : 0; u.w4c = (int)lround(f->w4_d / f->h);
    if (c->ref && c->uw_rep_ref > 0) { u.anchor = (uint32_t)np; u.wref = c->uw_rep_ref; }
    u.pass = 1; atomic_store(&u.next, 0); fit_zpar(x->nt, x->nt, uwr_tiles, &u);
    size_t nv = 0; for (int t = 0; t < x->nt; t++) nv += u.nv[t];
    l1_vote *v = malloc((nv ? nv : 1) * sizeof *v); nv = 0;
    for (int t = 0; t < x->nt; t++) { if (u.nv[t]) memcpy(v + nv, u.v[t], u.nv[t] * sizeof *v); nv += u.nv[t]; free(u.v[t]); }
    uint32_t nn = (uint32_t)np + (u.wref > 0); size_t na = 0;   /* the seed anchor: node np, tied to each piece holding seeded voxels (shift 0 wanted) */
    if (c->seed && c->uw_rep_seed > 0 && np) {
        int64_t *aw = calloc(np, sizeof *aw);
        if (aw) {
            for (int z = 0; z < g->D; z++) {
                const double pz = (z + 0.5) * f->h; if (pz >= c->seed_z || pz < c->seed_z0) continue;
                for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) { const uint32_t i = ub_id(g, z, y, xx); if (u.comp[i]) aw[uwr_gid(&u, z, y, xx, i)] += x->V[i].bk; }
            }
            for (uint64_t k = 0; k < np; k++) na += aw[k] > 0;
            l1_vote *v2 = na ? realloc(v, (nv + na) * sizeof *v) : v;
            if (v2) { v = v2; for (uint64_t k = 0; k < np; k++) if (aw[k] > 0) v[nv++] = (l1_vote){(uint32_t)k, (uint32_t)np, 0, (int64_t)llround(c->uw_rep_seed * (double)aw[k])}; nn = (uint32_t)np + 1; }
            else na = 0;
            free(aw);
        }
    }
    nv = l1_votes_merge(v, nv);
    {   /* UFSM_L1_DUMP=PATH: the solve's input (n, nv, then the merged votes) for ufsm l1-bench */
        const char *dp = getenv("UFSM_L1_DUMP");
        if (dp && *dp) { FILE *fo = fopen(dp, "wb"); const uint64_t n64 = nn, nv64 = nv; int bad = !fo;
            if (fo) { bad |= fwrite(&n64, 8, 1, fo) != 1 || fwrite(&nv64, 8, 1, fo) != 1 || (nv && fwrite(v, sizeof *v, nv, fo) != nv); bad |= fclose(fo) != 0; }
            if (bad) fprintf(stderr, "fit: cannot write the repair dump %s\n", dp); else fprintf(stderr, "fit: repair votes dumped to %s\n", dp);
            if (getenv("UFSM_L1_DUMP_EXIT")) exit(bad); }
    }
    const double t1 = nn_wall_seconds();
    int32_t *s = calloc(nn ? nn : 1, sizeof *s); l1_stats st = {0};
    if (u.wref > 0 && s) {   /* start each piece at its anchor votes' weighted median (the ref's integers): the flood's count can sit many
                                turns off the ref, and +-1 moves from 0 would take as many max flows (the merged votes: sorted by a, b, d) */
        size_t i = 0, ns0 = 0;
        while (i < nv) {
            const uint32_t a = v[i].a; size_t j = i; int64_t tw = 0;
            while (j < nv && v[j].a == a) { if (v[j].b == u.anchor) tw += v[j].w; j++; }
            if (tw > 0 && a < np) {
                int64_t cw = 0;
                for (size_t q = i; q < j; q++) if (v[q].b == u.anchor) { cw += v[q].w; if (2 * cw >= tw) { s[a] = -v[q].d; ns0 += s[a] != 0; break; } }
            }
            i = j;
        }
        if (c->verbose) fprintf(stderr, "fit: slip repair: %zu of %llu pieces start at the ref's integers\n", ns0, (unsigned long long)np);
    }
    if (c->ref_snap == 2 && u.wref > 0) { if (c->verbose) fprintf(stderr, "fit: slip repair: --ref-snap 2: each piece at its ref median, no L1 moves\n"); }
    else if (l1_solve(nn, v, nv, s, 256, &st)) fprintf(stderr, "fit: out of memory for the slip repair's solve (skipped)\n");
    if (st.capped) fprintf(stderr, "fit: slip repair: the max-flow work budget stopped the moves after %d max flows (%d moves, L1 %lld -> %lld, %.1fs); the last move's integers kept\n", st.iters, st.moves, (long long)st.e0, (long long)st.e1, st.secs);
    if (nn > np) { const int32_t s0 = s[np]; for (uint64_t k = 0; k < np; k++) s[k] -= s0; }   /* the anchor's gauge */
    if (c->verbose && na) fprintf(stderr, "fit: slip repair: %zu pieces anchored to the seed model's integers (weight %g)\n", na, (double)c->uw_rep_seed);
    const double t2 = nn_wall_seconds();
    size_t pm = 0, vm = 0;
    for (uint64_t k = 0; k < np; k++) pm += s[k] != 0;
    if (pm) {   /* the shifts (by z range), then the seams again */
        u.shift = s; fit_zpar(g->D, x->nt, uwr_apply, &u); u.shift = nullptr;
    }
    if (u.seam) { u.seam_shift = 4; }
    fit_zpar(g->D, x->nt, uwr_apply, &u);
    size_t seam1 = 0; for (int t = 0; t < x->nt; t++) seam1 += u.nseam[t];
    if (c->uw_rep_dump) {   /* PREFIX.shift.raw, PREFIX.seam.raw */
        uint8_t *sh = calloc(nf, 1);
        for (int z = 0; z < g->D; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) {
            const uint32_t i = ub_id(g, z, y, xx); if (!u.comp[i]) continue;
            const int32_t d = s[uwr_gid(&u, z, y, xx, i)]; sh[((size_t)z * g->H + y) * g->W + xx] = (uint8_t)(128 + (d < -127 ? -127 : d > 127 ? 127 : d));
        }
        char p[1200]; int bad = 0;
        snprintf(p, sizeof p, "%s.shift.raw", c->uw_rep_dump); FILE *fo = fopen(p, "wb"); bad |= !fo || fwrite(sh, 1, nf, fo) != nf; if (fo) bad |= fclose(fo) != 0;
        snprintf(p, sizeof p, "%s.seam.raw", c->uw_rep_dump); fo = fopen(p, "wb"); bad |= !fo || fwrite(u.seam, 1, nf, fo) != nf; if (fo) bad |= fclose(fo) != 0;
        if (bad) fprintf(stderr, "fit: cannot write the repair dump %s.*\n", c->uw_rep_dump);
        free(sh);
    }
    for (int z = 0; z < g->D; z++) for (int y = 0; y < g->H; y++) for (int xx = 0; xx < g->W; xx++) { const uint32_t i = ub_id(g, z, y, xx); if (u.comp[i] && s[uwr_gid(&u, z, y, xx, i)]) vm++; }
    if (c->verbose) fprintf(stderr, "fit: slip repair (tiles %d^3): %llu pieces, %zu edges (%zu votes), L1 disagreement %lld -> %lld in %d moves (%d max flows, %.1fs); "
                            "%zu pieces / %zu voxels shifted; seam pairs %zu -> %zu; pieces + votes %.1fs\n",
                            T, (unsigned long long)np, st.nedge, nv, (long long)st.e0, (long long)st.e1, st.moves, st.iters, t2 - t1, pm, vm, seam0 / 2, seam1 / 2, t1 - t0);
    free(v); free(s); free(u.comp); free(u.npc); free(u.tbase); free(u.seam); free(u.bad); free(u.lst);
}
/* the consensus repair (fit_cfg uw_rep_cons K > 1): K repairs of the same flood on tile grids offset by i T / K (+ uw_rep_off), each
   voxel's integer the most frequent of the K after aligning every run's gauge to the first (the mode of their per-voxel difference);
   ties keep the first run's. One L1 solve's choice of where to cut a dislocation moves with the tiling; a cut that the tilings agree on
   survives the vote. */
static void uw_repair(uw_ctx *x, size_t nr, const fit_cfg *c) {
    const int K = c->uw_rep_cons, T = c->uw_repair;
    if (K <= 1) { uw_repair1(x, nr, c, c->uw_rep_off); return; }
    uw_vox *V = x->V; int16_t *k0 = malloc(nr * sizeof *k0), *ks = malloc(nr * (size_t)K * sizeof *ks);
    if (!k0 || !ks) { free(k0); free(ks); fprintf(stderr, "fit: no memory for the consensus repair (one repair)\n"); uw_repair1(x, nr, c, c->uw_rep_off); return; }
    for (size_t i = 0; i < nr; i++) k0[i] = V[i].k;
    for (int r = 0; r < K; r++) {
        if (r) for (size_t i = 0; i < nr; i++) V[i].k = k0[i];
        uw_repair1(x, nr, c, c->uw_rep_off + (int)((long)r * T / K));
        int16_t *o = ks + (size_t)r * nr; for (size_t i = 0; i < nr; i++) o[i] = V[i].k;
    }
    size_t moved = 0, split = 0;
    for (int r = 1; r < K; r++) {   /* the gauge of run r against run 0 */
        size_t h[65] = {0}; int16_t *o = ks + (size_t)r * nr;
        for (size_t i = 0; i < nr; i++) { if (k0[i] == UK_UNSET) continue; const int d = o[i] - ks[i]; if (d >= -32 && d <= 32) h[d + 32]++; }
        int m = 32; for (int d = 0; d < 65; d++) if (h[d] > h[m]) m = d;
        if (m != 32) for (size_t i = 0; i < nr; i++) if (k0[i] != UK_UNSET) o[i] = (int16_t)(o[i] - (m - 32));
    }
    for (size_t i = 0; i < nr; i++) {
        if (k0[i] == UK_UNSET) continue;
        int best = ks[i], bn = 0;
        for (int a = 0; a < K; a++) { const int v = ks[(size_t)a * nr + i]; int n = 0; for (int b = 0; b < K; b++) n += ks[(size_t)b * nr + i] == v; if (n > bn) { bn = n; best = v; } }
        if (2 * bn <= K) split++;
        moved += best != ks[i]; V[i].k = (int16_t)best;
    }
    if (c->verbose) fprintf(stderr, "fit: consensus repair over %d tilings: %zu voxels moved off the first tiling's integers, %zu without a majority\n", K, moved, split);
    free(k0); free(ks);
}

/* Parallel voxel unwrapping (fit_cfg uw_msf): the serial flood grows a quality-ordered spanning tree from the seed (each voxel takes
   its integer from its first popped neighbour); here the same kind of tree is built in parallel as the maximum spanning forest of the
   unwrappable voxels' 6-neighbour graph under a total edge order (weight = the lower confidence level of the two voxels, then the
   higher, then the lower edge id): Boruvka rounds (each component picks its best outgoing edge, components hook along them, pointer
   jumping), integers carried as union-find potentials (k_b = k_a + round(uw_a - uw_b) along every tree edge, the flood's step). The
   forest is unique for a total order, so the result does not depend on threads or timing. It equals the flood's tree up to the
   order among equal confidence levels (the flood: first popped, FIFO; here: the higher partner level, then the edge id), which only
   matters where such ties close a loop around a dislocation. The seed's component gets the integers, the rest stays unreached, as
   the flood. Memory: ~20 B per unwrappable voxel (compact ids). */
typedef struct {
    uw_ctx *x; size_t nb;                 /* bricks of 512 records */
    uint64_t *mask; uint16_t *pre; uint32_t *base;   /* per brick: valid-record bits, cumulative counts per 64-bit word, first compact id */
    uint32_t *rec;                        /* compact id -> record id */
    uint32_t *par; int32_t *pot;          /* union-find over compact ids (par: the root after each round; pot: k_v - k_root) */
    uint32_t *par2; int32_t *pot2;        /* (pointer jumping: double buffers) */
    _Atomic uint64_t *best;               /* per root: the best outgoing edge key (0 none) */
    uint32_t *roots; size_t nroots;       /* the current roots (compact ids) */
    _Atomic size_t hooks; size_t nv;
    int moved[64];                        /* msf_jump: per thread, a pointer changed */
    int collect;                          /* msf_edges: also keep the edges between components (per thread), later rounds scan only those */
    struct msf_e { uint64_t key; uint32_t cu, cv; } *el[64], *E; size_t eln[64], elc[64], ne;
} msf_ctx;
static inline uint32_t msf_cid(const msf_ctx *M, uint32_t id) {
    const size_t b = id >> 9; const unsigned r = id & 511, w = r >> 6, bit = r & 63;
    return M->base[b] + M->pre[b * 8 + w] + (uint32_t)__builtin_popcountll(M->mask[b * 8 + w] & ((1ull << bit) - 1));
}
static void msf_mask(void *a, int b0, int b1, int t) {   /* bricks [b0, b1): valid bits and counts */
    msf_ctx *M = a; const uw_vox *V = M->x->V;
    for (size_t b = (size_t)b0; b < (size_t)b1; b++) {
        unsigned c = 0;
        for (int w = 0; w < 8; w++) {
            uint64_t m = 0; for (int k = 0; k < 64; k++) m |= (uint64_t)(V[(b << 9) + w * 64 + k].bk != 0) << k;
            M->mask[b * 8 + w] = m; M->pre[b * 8 + w] = (uint16_t)c; c += (unsigned)__builtin_popcountll(m);
        }
        M->base[b] = c;   /* (the count; prefix-summed after) */
    }
}
static void msf_init(void *a, int b0, int b1, int t) {   /* compact ids: records, every node its own root */
    msf_ctx *M = a;
    for (size_t b = (size_t)b0; b < (size_t)b1; b++) for (int w = 0; w < 8; w++) {
        uint64_t m = M->mask[b * 8 + w];
        while (m) { const int k = __builtin_ctzll(m); m &= m - 1; const uint32_t id = (uint32_t)((b << 9) + w * 64 + k), c = msf_cid(M, id); M->rec[c] = id; M->par[c] = c; M->pot[c] = 0; }
    }
}
static inline uint64_t msf_key(unsigned lo, unsigned hi, uint64_t eid) { return (uint64_t)lo << 46 | (uint64_t)hi << 34 | ((((uint64_t)1 << 34) - 1) - eid); }
static inline void msf_amax(_Atomic uint64_t *p, uint64_t v) { uint64_t o = atomic_load_explicit(p, memory_order_relaxed); while (o < v && !atomic_compare_exchange_weak_explicit(p, &o, v, memory_order_relaxed, memory_order_relaxed)) {} }
/* the neighbour of record id along axis ax (0 z, 1 y, 2 x; +1), or UINT32_MAX outside the box or not unwrappable */
static inline uint32_t msf_nbr(const msf_ctx *M, uint32_t id, int ax) {
    const uw_geo *g = &M->x->g; int z, y, xx; ub_zyx(g, id, &z, &y, &xx);
    if (ax == 0) { if (++z >= g->D) return UINT32_MAX; } else if (ax == 1) { if (++y >= g->H) return UINT32_MAX; } else { if (++xx >= g->W) return UINT32_MAX; }
    const uint32_t j = ub_id(g, z, y, xx); return M->x->V[j].bk ? j : UINT32_MAX;
}
static void msf_edges(void *a, int b0, int b1, int t) {   /* every edge between two components: both roots' best */
    msf_ctx *M = a; const uw_vox *V = M->x->V;
    for (size_t b = (size_t)b0; b < (size_t)b1; b++) for (int w = 0; w < 8; w++) {
        uint64_t m = M->mask[b * 8 + w];
        while (m) {
            const int k = __builtin_ctzll(m); m &= m - 1; const uint32_t id = (uint32_t)((b << 9) + w * 64 + k), cu = msf_cid(M, id), ru = M->par[cu];
            for (int ax = 0; ax < 3; ax++) {
                const uint32_t j = msf_nbr(M, id, ax); if (j == UINT32_MAX) continue;
                const uint32_t cv = msf_cid(M, j), rv = M->par[cv]; if (ru == rv) continue;
                const unsigned bu = V[id].bk, bv = V[j].bk;
                const uint64_t key = msf_key(bu < bv ? bu : bv, bu < bv ? bv : bu, (uint64_t)cu * 3 + ax);
                msf_amax(&M->best[ru], key); msf_amax(&M->best[rv], key);
                if (M->collect) {
                    if (M->eln[t] == M->elc[t]) { M->elc[t] = M->elc[t] ? 2 * M->elc[t] : 4096; struct msf_e *n = realloc(M->el[t], M->elc[t] * sizeof *n); if (!n) { M->collect = -1; continue; } M->el[t] = n; }
                    M->el[t][M->eln[t]++] = (struct msf_e){key, cu, cv};
                }
            }
        }
    }
}
static void msf_edges_list(void *a, int e0, int e1, int t) {   /* the collected edges: those still between two components */
    msf_ctx *M = a;
    for (size_t e = (size_t)e0; e < (size_t)e1; e++) {
        const struct msf_e *E = &M->E[e]; const uint32_t ru = M->par[E->cu], rv = M->par[E->cv]; if (ru == rv) continue;
        msf_amax(&M->best[ru], E->key); msf_amax(&M->best[rv], E->key);
    }
}
static inline double msf_d(const uw_vox *V, uint32_t a, uint32_t b) { return round((double)V[a].uw - (double)V[b].uw); }   /* k_b - k_a */
static void msf_hook(void *a, int i0, int i1, int t) {   /* each root with an edge: its parent root and potential (mutual best: the lower root stays) */
    msf_ctx *M = a; const uw_vox *V = M->x->V; size_t nh = 0;
    for (size_t i = (size_t)i0; i < (size_t)i1; i++) {
        const uint32_t r = M->roots[i]; const uint64_t key = atomic_load_explicit(&M->best[r], memory_order_relaxed);
        M->par2[r] = r; M->pot2[r] = 0;
        if (!key) continue;
        const uint64_t eid = (((uint64_t)1 << 34) - 1) - (key & (((uint64_t)1 << 34) - 1)); const uint32_t cu = (uint32_t)(eid / 3); const int ax = (int)(eid % 3);
        const uint32_t iu = M->rec[cu], iv = msf_nbr(M, iu, ax), cv = msf_cid(M, iv);
        const uint32_t ru = M->par[cu], rv = M->par[cv], s = ru == r ? rv : ru;   /* the other side's root */
        if (atomic_load_explicit(&M->best[s], memory_order_relaxed) == key && r < s) continue;   /* mutual: r stays a root */
        /* k_r - k_s from the edge: k_iv = k_iu + d(iu, iv); k_x = k_root(x) + pot(x) */
        const int32_t d = (int32_t)msf_d(V, iu, iv);
        M->par2[r] = s; M->pot2[r] = ru == r ? M->pot[cv] - M->pot[cu] - d : M->pot[cu] - M->pot[cv] + d;
        nh++;
    }
    atomic_fetch_add(&M->hooks, nh);
}
static void msf_jump(void *a, int i0, int i1, int t) {   /* one pointer-jumping step over the roots: par2 / pot2 -> par / pot (as scratch) */
    msf_ctx *M = a; int mv = 0;
    for (size_t i = (size_t)i0; i < (size_t)i1; i++) {
        const uint32_t r = M->roots[i], p = M->par2[r];
        M->par[r] = M->par2[p]; M->pot[r] = M->pot2[r] + (p != r ? M->pot2[p] : 0); mv |= M->par2[p] != p;
    }
    M->moved[t] = mv;
}
static void msf_copyback(void *a, int i0, int i1, int t) {   /* par / pot -> par2 / pot2 (the next jumping step reads them) */
    msf_ctx *M = a; for (size_t i = (size_t)i0; i < (size_t)i1; i++) { const uint32_t r = M->roots[i]; M->par2[r] = M->par[r]; M->pot2[r] = M->pot[r]; }
}
/* every node c: r = par[c] its round-start root (or, for a hooked root, already its final root, whose par2 / pot2 are itself / 0)
   -> the final root par2[r], potential + pot2[r] (reads only the roots' par2 / pot2, writes only c's own entries) */
static void msf_relabel(void *a, int c0, int c1, int t) {
    msf_ctx *M = a;
    for (size_t c = (size_t)c0; c < (size_t)c1; c++) { const uint32_t r = M->par[c]; M->pot[c] += M->pot2[r]; M->par[c] = M->par2[r]; }
}
static size_t uw_msf(uw_ctx *x, size_t nr, uint32_t sid, int32_t kseed, int verbose) {
    msf_ctx M = {.x = x, .nb = nr >> 9}; const double t0 = nn_wall_seconds();
    M.mask = malloc(M.nb * 8 * sizeof *M.mask); M.pre = malloc(M.nb * 8 * sizeof *M.pre); M.base = malloc((M.nb + 1) * sizeof *M.base);
    if (!M.mask || !M.pre || !M.base) { free(M.mask); free(M.pre); free(M.base); return (size_t)-1; }
    fit_zpar((int)M.nb, fit_nthreads((int)M.nb), msf_mask, &M);
    size_t nv = 0; for (size_t b = 0; b < M.nb; b++) { const uint32_t c = M.base[b]; M.base[b] = (uint32_t)nv; nv += c; }
    M.base[M.nb] = (uint32_t)nv; M.nv = nv;
    if (nv >= ((uint64_t)1 << 32) / 3) { free(M.mask); free(M.pre); free(M.base); return (size_t)-1; }
    M.rec = malloc(nv * 4); M.par = malloc(nv * 4); M.pot = malloc(nv * 4); M.par2 = malloc(nv * 4); M.pot2 = malloc(nv * 4); M.best = malloc(nv * 8); M.roots = malloc(nv * 4);
    if (!M.rec || !M.par || !M.pot || !M.par2 || !M.pot2 || !M.best || !M.roots) { free(M.rec); free(M.par); free(M.pot); free(M.par2); free(M.pot2); free(M.best); free(M.roots); free(M.mask); free(M.pre); free(M.base); return (size_t)-1; }
    fit_zpar((int)M.nb, fit_nthreads((int)M.nb), msf_init, &M);
    for (size_t c = 0; c < nv; c++) M.roots[c] = (uint32_t)c;
    M.nroots = nv; int rounds = 0;
    while (M.nroots > 1) {
        for (size_t i = 0; i < M.nroots; i++) atomic_store_explicit(&M.best[M.roots[i]], 0, memory_order_relaxed);
        /* once under 1 / 256 of the nodes are roots: only the edges still between components (collected by that scan, filtered each round;
           the same keys, so the same forest) */
        const double te0 = nn_wall_seconds();
        if (M.E) fit_zpar((int)M.ne, fit_nthreads((int)M.ne), msf_edges_list, &M);
        else {
            M.collect = M.nroots * 256 < M.nv;   /* (late: after round 1 the boundary edges of 7 M components were 4.6 s and 6 GB on a slab) */
            fit_zpar((int)M.nb, fit_nthreads((int)M.nb), msf_edges, &M);
            if (M.collect == 1) {
                size_t ne = 0; for (int t = 0; t < 64; t++) ne += M.eln[t];
                M.E = malloc((ne ? ne : 1) * sizeof *M.E);
                if (M.E) { for (int t = 0; t < 64; t++) if (M.eln[t]) { memcpy(M.E + M.ne, M.el[t], M.eln[t] * sizeof *M.E); M.ne += M.eln[t]; } }
            }
            for (int t = 0; t < 64; t++) { free(M.el[t]); M.el[t] = nullptr; M.eln[t] = M.elc[t] = 0; }
            M.collect = 0;
        }
        const double te1 = nn_wall_seconds();
        atomic_store(&M.hooks, 0);
        const int nri = (int)M.nroots, ntr = fit_nthreads(nri);
        fit_zpar(nri, ntr, msf_hook, &M);
        rounds++;
        if (!atomic_load(&M.hooks)) break;
        /* pointer jumping over the roots until every root points at a final root (par2 / pot2 hold the current pointers) */
        for (int it = 0; it < 64; it++) {
            fit_zpar(nri, ntr, msf_jump, &M);
            int moved = 0; for (int t = 0; t < ntr; t++) moved |= M.moved[t];
            fit_zpar(nri, ntr, msf_copyback, &M);
            if (!moved) break;
        }
        /* the non-roots follow their roots (roots already hold their final root and potential in par / pot = par2 / pot2) */
        fit_zpar((int)nv, fit_nthreads((int)nv), msf_relabel, &M);
        size_t k = 0; for (size_t i = 0; i < M.nroots; i++) { const uint32_t r = M.roots[i]; if (M.par[r] == r) M.roots[k++] = r; }
        if (getenv("UFSM_FIT_TIMES")) fprintf(stderr, "fit times: msf round %d: %zu -> %zu roots, edges %.2fs, rest %.2fs\n", rounds, M.nroots, k, te1 - te0, nn_wall_seconds() - te1);
        M.nroots = k;
        if (M.E) { size_t j = 0; for (size_t e = 0; e < M.ne; e++) if (M.par[M.E[e].cu] != M.par[M.E[e].cv]) M.E[j++] = M.E[e]; M.ne = j; }   /* (the order kept) */
    }
    free(M.E);
    /* the integers: the seed's component, k_seed fixed */
    const uint32_t cs = msf_cid(&M, sid), R = M.par[cs]; const int32_t kR = kseed - M.pot[cs];
    size_t reached = 0, over = 0; uw_vox *V = x->V;
    for (size_t c = 0; c < nv; c++) {
        if (M.par[c] != R) continue;
        const int64_t k = (int64_t)kR + M.pot[c];
        if (k <= UK_UNSET || k > INT16_MAX) { over++; continue; }
        V[M.rec[c]].k = (int16_t)k; reached++;
    }
    if (over) atomic_fetch_add(&x->over, over);
    if (verbose) fprintf(stderr, "fit: voxel unwrapping (spanning forest): %zu voxels, %d rounds, %.1fs\n", nv, rounds, nn_wall_seconds() - t0);
    free(M.rec); free(M.par); free(M.pot); free(M.par2); free(M.pot2); free(M.best); free(M.roots); free(M.mask); free(M.pre); free(M.base);
    return reached;
}
typedef struct { const fit_model *m; uint8_t *have; double *corr; int changed[64]; } nf_job;
static void uw_nodefill(void *a, int a0, int a1, int t) {   /* lattice z planes [a0, a1): one pass of the node fill */
    nf_job *J = a; const fit_model *m = J->m; int changed = 0;
    for (size_t k = (size_t)a0 * m->ly * m->lx; k < (size_t)a1 * m->ly * m->lx; k++) {
        if (J->have[k]) continue;
        const int aa = (int)(k / ((size_t)m->ly * m->lx)), b = (int)((k / m->lx) % m->ly), cc = (int)(k % m->lx);
        double sm = 0; int cnt = 0;
        const int nb[6][3] = {{aa - 1, b, cc}, {aa + 1, b, cc}, {aa, b - 1, cc}, {aa, b + 1, cc}, {aa, b, cc - 1}, {aa, b, cc + 1}};
        for (int q = 0; q < 6; q++) {
            if (nb[q][0] < 0 || nb[q][1] < 0 || nb[q][2] < 0 || nb[q][0] >= m->lz || nb[q][1] >= m->ly || nb[q][2] >= m->lx) continue;
            const size_t j = ((size_t)nb[q][0] * m->ly + nb[q][1]) * m->lx + nb[q][2];
            if (J->have[j] == 1) { sm += J->corr[j]; cnt++; }
        }
        if (cnt) { J->corr[k] = sm / cnt; J->have[k] = 2; changed = 1; }
    }
    J->changed[t] = changed;
}
static size_t fit_unwrap_voxels(fit_model *m, const fit_fields *f, double min_conf, const fit_cfg *c) {
    const size_t n = (size_t)f->D * f->H * f->W; const int na = (int)(f->D * f->h) + 1;
    const uw_geo g = {f->D, f->H, f->W, (f->H + UB_M) / UB_S, (f->W + UB_M) / UB_S};
    const size_t nr = ((size_t)((f->D + UB_M) / UB_S) * g.bH * g.bW) << (3 * UB);   /* records, the bricks padded */
    if (nr >= UQ_NONE) { fprintf(stderr, "fit: %zu field voxels exceed the unwrapping's 32-bit queue: use --slab\n", n); return 0; }
    const double t0 = nn_wall_seconds();
    uw_ctx x = {.m = m, .f = f, .c = c, .min_conf = min_conf, .na = na, .nt = fit_nthreads(f->D), .g = g};
    pthread_once(&g_uw_ang_once, uw_ang_init);
    void *vbase = nullptr; size_t vlen = 0; uw_vox *V = x.V = uw_alloc_records(nr * sizeof *V, &vbase, &vlen);   /* (zero: every record UK_UNSET, bk 0) */
    if (!V) { fprintf(stderr, "fit: out of memory for the voxel unwrapping\n"); return 0; }
    fit_zpar(f->D, x.nt, uw_setup, &x);
    size_t seed = (size_t)-1; float qbest = 0;
    for (int t = 0; t < x.nt; t++) if (x.qb[t] > qbest) { qbest = x.qb[t]; seed = x.sb[t]; }   /* ties: the lowest voxel, as serially */
    const double t1 = nn_wall_seconds();
    size_t reached = 0;
    uw_queue q; if (uq_init(&q)) { munmap(vbase, vlen); return 0; }
    if (seed != (size_t)-1) {
        {   /* the integer base: the seed (or seed model's) integer at the most confident voxel */
            const int z = (int)(seed / ((size_t)f->H * f->W)), y = (int)((seed / f->W) % f->H), xx = (int)(seed % f->W);
            const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
            const double th = atan2(py - axis_lerp(f->cy, na, pz), px - axis_lerp(f->cx, na, pz));
            if (c->ref) {   /* the sub-turn offset between the ref and the phase */
                fit_zpar(f->D, x.nt, uw_refbias, &x);
                double complex s = 0; double w = 0; for (int t = 0; t < x.nt; t++) { s += x.rph[t]; w += x.rpw[t]; }
                x.rbias = cabs(s) > 0 ? carg(s) / (2 * M_PI) : 0;
                if (c->verbose) fprintf(stderr, "fit: ref model: sub-turn offset U_ref - phase %+.3f turn (coherence %.3f)%s\n", x.rbias, w > 0 ? cabs(s) / w : 0,
                                        c->ref_snap ? ", integers snapped to the ref" : "");
            }
            const double Ub = (c->ref ? uw_Uref(&x, z, y, xx) - x.rbias : (c->seed ? fit_W(c->seed, pz + c->seed_off, py, px) : fit_W(m, pz, py, px)) + m->dir * th / (2 * M_PI));
            x.kb = (int32_t)round(Ub - V[ub_id(&g, z, y, xx)].uw) - UK_BIAS;
        }
        if (c->ref && c->ref_snap == 1) {   /* the ref's integers everywhere, no flood */
            fit_zpar(f->D, x.nt, uw_refsnap, &x);
            for (int t = 0; t < x.nt; t++) reached += x.nreach[t];   /* (the records with bk and an integer) */
        }
        else if (c->uw_split > 0 && !c->seed && !c->ref) reached = uw_two_level(&x, c->uw_split, c->uw_tile > 0 ? c->uw_tile : 256, nr, c->verbose);
        else {
            if (c->seed) {
                fit_zpar(f->D, x.nt, uw_seedmodel, &x);
                for (int z = 0; z < f->D; z++) for (int y = 0; y < f->H; y++) for (int xx = 0; xx < f->W; xx++) {   /* linear order, as before */
                    const uint32_t id = ub_id(&g, z, y, xx); if (V[id].bk && V[id].k != UK_UNSET) { reached++; uq_push(&q, id, V[id].bk); }
                }
            }
            size_t rm = (size_t)-1;
            if (!reached) {
                const int z = (int)(seed / ((size_t)f->H * f->W)), y = (int)((seed / f->W) % f->H), xx = (int)(seed % f->W);
                const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (xx + 0.5) * f->h;
                const double th = atan2(py - axis_lerp(f->cy, na, pz), px - axis_lerp(f->cx, na, pz));
                const double Um = c->ref ? uw_Uref(&x, z, y, xx) - x.rbias : fit_W(m, pz, py, px) + m->dir * th / (2 * M_PI);
                const uint32_t sid = ub_id(&g, z, y, xx); const int32_t ks = (int32_t)round(Um - V[sid].uw) - x.kb;
                if (c->uw_msf && !c->seed) rm = uw_msf(&x, nr, sid, ks, c->verbose);   /* the parallel spanning forest (seeded fits: the flood) */
                if (rm == (size_t)-1) { V[sid].k = (int16_t)ks; reached = 1; uq_push(&q, sid, V[sid].bk); }
            }
            if (rm != (size_t)-1) reached = rm;
            else reached += uw_flood(&x, &q, &(uw_box){0, f->D, 0, f->H, 0, f->W, 1, nullptr, 0});
        }
    }
    uq_free(&q);
    if (c->uw_repair > 0 && reached) uw_repair(&x, nr, c);
    const double t2 = nn_wall_seconds();
    /* per lattice node: mean of (U_unwrapped - U_model) over the reached voxels nearest to it */
    const size_t nl = (size_t)m->lz * m->ly * m->lx;
    x.acc = calloc(nl, sizeof(double)); x.wsum = calloc(nl, sizeof(double));
    {   /* the node rows' field rows: z rows with lround(pz / L) (clamped) == az, y rows likewise per node row bb */
        int *zr = calloc(m->lz + 1, sizeof(int)), *yr = calloc(m->ly + 1, sizeof(int));
        for (int z = 0; z < f->D; z++) { int az = (int)lround((z + 0.5) * f->h / m->L); if (az >= m->lz) az = m->lz - 1; zr[az + 1]++; }
        for (int y = 0; y < f->H; y++) { int b = (int)lround((y + 0.5) * f->h / m->L); if (b >= m->ly) b = m->ly - 1; yr[b + 1]++; }
        for (int a = 0; a < m->lz; a++) zr[a + 1] += zr[a];
        for (int b = 0; b < m->ly; b++) yr[b + 1] += yr[b];
        ua_job J = {&x, zr, yr}; fit_par(m->lz * m->ly, uw_accum, &J);
        free(zr); free(yr);
    }
    uint8_t *have = calloc(nl, 1); double *corr = calloc(nl, sizeof(double));
    for (size_t k = 0; k < nl; k++) if (x.wsum[k] > 0) { corr[k] = x.acc[k] / x.wsum[k]; have[k] = 1; }
    for (int pass = 0; pass < 64; pass++) {   /* nodes without voxels: the mean of their neighbours (a pass reads only earlier passes'
                                                 nodes, have 1, and writes only have-0 nodes: lattice z planes on threads, the serial result) */
        nf_job J = {m, have, corr, {0}};
        fit_zpar(m->lz, fit_nthreads(m->lz), uw_nodefill, &J);
        int changed = 0; for (int t = 0; t < 64; t++) changed |= J.changed[t];
        for (size_t k = 0; k < nl; k++) if (have[k] == 2) have[k] = 1;
        if (!changed) break;
    }
    for (size_t k = 0; k < nl; k++) m->delta[k] += (float)corr[k];
    if (c->verbose) fprintf(stderr, "fit: voxel unwrapping %d threads: setup %.1fs, flood %.1fs, nodes %.1fs\n", x.nt, t1 - t0, t2 - t1, nn_wall_seconds() - t2);
    if (x.over) fprintf(stderr, "fit: %zu voxels over 16383 turns from the seed left unreached\n", (size_t)x.over);
    munmap(vbase, vlen); free(x.acc); free(x.wsum); free(have); free(corr);
    return reached;
}

static void fit_unwrap_init(fit_model *m, const fit_fields *f, double min_coh) {
    const size_t nl = (size_t)m->lz * m->ly * m->lx, n = (size_t)f->D * f->H * f->W;
    double complex *A = calloc(nl, sizeof *A); double *S = calloc(nl, sizeof(double));
    for (size_t i = 0; i < n; i++) {
        if (!f->w[i]) continue;
        const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0, conf = sqrt(mc * mc + ms * ms);
        if (conf < 0.05) continue;
        const int z = (int)(i / ((size_t)f->H * f->W)), y = (int)((i / f->W) % f->H), x = (int)(i % f->W);
        const double pz = (z + 0.5) * f->h, py = (y + 0.5) * f->h, px = (x + 0.5) * f->h;
        const double W = fit_W(m, pz, py, px), w = f->w[i] / 255.0 * conf;
        const int a = (int)lround(pz / m->L), b = (int)lround(py / m->L), c = (int)lround(px / m->L);
        const size_t k = ((size_t)(a < m->lz ? a : m->lz - 1) * m->ly + (b < m->ly ? b : m->ly - 1)) * m->lx + (c < m->lx ? c : m->lx - 1);
        A[k] += w * cexp(I * (atan2(ms, mc) - 2 * M_PI * W)); S[k] += w;
    }
    double *corr = calloc(nl, sizeof(double)); uint8_t *vis = calloc(nl, 1); fit_heap_e *hp = malloc(nl * 7 * sizeof *hp); int hn = 0;
    size_t seed = 0; for (size_t k = 1; k < nl; k++) if (cabs(A[k]) > cabs(A[seed])) seed = k;
    if (S[seed] > 0) { corr[seed] = carg(A[seed]) / (2 * M_PI); vis[seed] = 1; heap_push(hp, &hn, (fit_heap_e){cabs(A[seed]), (int)seed}); }
    while (hn) {
        const fit_heap_e e = heap_pop(hp, &hn); const int k = e.i;
        const int a = k / (m->ly * m->lx), b = (k / m->lx) % m->ly, c = k % m->lx;
        const int nb[6][3] = {{a - 1, b, c}, {a + 1, b, c}, {a, b - 1, c}, {a, b + 1, c}, {a, b, c - 1}, {a, b, c + 1}};
        for (int q = 0; q < 6; q++) {
            if (nb[q][0] < 0 || nb[q][1] < 0 || nb[q][2] < 0 || nb[q][0] >= m->lz || nb[q][1] >= m->ly || nb[q][2] >= m->lx) continue;
            const size_t j = ((size_t)nb[q][0] * m->ly + nb[q][1]) * m->lx + nb[q][2];
            if (vis[j] || S[j] <= 0 || cabs(A[j]) < min_coh * S[j]) continue;
            double v = carg(A[j]) / (2 * M_PI); v += round(corr[k] - v);
            corr[j] = v; vis[j] = 1; heap_push(hp, &hn, (fit_heap_e){cabs(A[j]), (int)j});
        }
    }
    for (int pass = 0; pass < 64; pass++) {   /* the rest: mean of visited neighbours, repeated until none is left */
        int changed = 0;
        for (size_t k = 0; k < nl; k++) {
            if (vis[k]) continue;
            const int a = (int)(k / ((size_t)m->ly * m->lx)), b = (int)((k / m->lx) % m->ly), c = (int)(k % m->lx);
            double s = 0; int cnt = 0;
            const int nb[6][3] = {{a - 1, b, c}, {a + 1, b, c}, {a, b - 1, c}, {a, b + 1, c}, {a, b, c - 1}, {a, b, c + 1}};
            for (int q = 0; q < 6; q++) {
                if (nb[q][0] < 0 || nb[q][1] < 0 || nb[q][2] < 0 || nb[q][0] >= m->lz || nb[q][1] >= m->ly || nb[q][2] >= m->lx) continue;
                const size_t j = ((size_t)nb[q][0] * m->ly + nb[q][1]) * m->lx + nb[q][2];
                if (vis[j] == 1) { s += corr[j]; cnt++; }
            }
            if (cnt) { corr[k] = s / cnt; vis[k] = 2; changed = 1; }
        }
        for (size_t k = 0; k < nl; k++) if (vis[k] == 2) vis[k] = 1;
        if (!changed) break;
    }
    for (size_t k = 0; k < nl; k++) m->delta[k] += (float)corr[k];
    free(A); free(S); free(corr); free(vis); free(hp);
}

/* fields pooled by f per axis for the gradient-only stages (no phase): material-weighted means of n n^T and of the density
   rho (spacing re-encoded for the coarser h), the material fraction; the phase vector is not pooled (pc = ps = 128). */
typedef struct { int D, H, W; uint8_t *a[10]; float *pg; uint8_t *cs; } fit_pooled;   /* pg (with_pg): Gz Gy Gx c planes, src/fit_terms.h;
                                                                                       cs: the CT sheet spacing pooled as sp (fields with cs) */
typedef struct { const fit_fields *f; int fp; fit_pooled *p; } pool_job;
static void fit_pool_rows(void *a, int r0, int r1, int t_);
static fit_pooled fit_pool(const fit_fields *f, int fp, int with_pg) {   /* output (z, y) rows on threads (each cell its own sums: exact) */
    fit_pooled p = {(f->D + fp - 1) / fp, (f->H + fp - 1) / fp, (f->W + fp - 1) / fp, {0}, nullptr, nullptr};
    const size_t np = (size_t)p.D * p.H * p.W;
    for (int k = 0; k < 10; k++) p.a[k] = calloc(np, 1);
    if (f->cs) p.cs = calloc(np, 1);
    if (with_pg) p.pg = calloc(4 * np, sizeof(float));
    pool_job J = {f, fp, &p};
    fit_zpar(p.D * p.H, fit_nthreads(p.D * p.H), fit_pool_rows, &J);
    return p;
}
static void fit_pool_rows(void *a_, int r0, int r1, int t_) {
    const pool_job *J = a_; const fit_fields *f = J->f; const int fp = J->fp; fit_pooled p = *J->p;
    const size_t np = (size_t)p.D * p.H * p.W;
    const size_t HWf = (size_t)f->H * f->W;
    for (int zy = r0; zy < r1; zy++) for (int z = zy / p.H, y = zy % p.H, x = 0; x < p.W; x++) {
        double sw = 0, st[6] = {0}, srho = 0, swr = 0, gs[3] = {0, 0, 0}, gc[3] = {0, 0, 0}, crho = 0, cwr = 0; int cnt = 0;
        for (int a = 0; a < fp && z * fp + a < f->D; a++) for (int b = 0; b < fp && y * fp + b < f->H; b++) for (int e = 0; e < fp && x * fp + e < f->W; e++) {
            const size_t i = ((size_t)(z * fp + a) * f->H + y * fp + b) * f->W + x * fp + e; cnt++;
            if (!f->w[i]) continue;
            if (p.pg) {   /* the wrapped phase steps to the next cells, confidence-weighted (as fit_pgrad_term at full resolution) */
                const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0, c0 = sqrt(mc * mc + ms * ms);
                const int pos[3] = {z * fp + a, y * fp + b, x * fp + e}, dim[3] = {f->D, f->H, f->W}; const size_t sd[3] = {HWf, (size_t)f->W, 1};
                if (c0 >= 0.05) for (int d = 0; d < 3; d++) {
                    if (pos[d] + 1 >= dim[d] || !f->w[i + sd[d]]) continue;
                    const double nc = (f->pc[i + sd[d]] - 128) / 127.0, ns = (f->ps[i + sd[d]] - 128) / 127.0, c1 = sqrt(nc * nc + ns * ns);
                    if (c1 < 0.05) continue;
                    double ph = (atan2(ns, nc) - atan2(ms, mc)) / (2 * M_PI); ph -= floor(ph + 0.5);
                    const double cw = fmin(fmin(c0, c1), 1.0); gs[d] += cw * ph / f->h; gc[d] += cw;
                }
            }
            const double w = f->w[i] / 255.0; sw += w;
            for (int k = 0; k < 6; k++) st[k] += w * f->t[k][i];
            if (f->sp[i]) { srho += w * 8.0 / (f->h * f->sp[i]); swr += w; }
            if (p.cs && f->cs[i]) { crho += w * 8.0 / (f->h * f->cs[i]); cwr += w; }
        }
        const size_t o = ((size_t)z * p.H + y) * p.W + x;
        p.a[0][o] = p.a[1][o] = 128;
        if (p.pg && cnt) {
            double cs = 0; for (int d = 0; d < 3; d++) { p.pg[d * np + o] = gc[d] > 0 ? (float)(gs[d] / gc[d]) : 0.f; cs += gc[d]; }
            p.pg[3 * np + o] = (float)(gc[0] > 0 && gc[1] > 0 && gc[2] > 0 ? cs / (3.0 * cnt) : 0);
        }
        if (sw <= 0) continue;
        p.a[9][o] = (uint8_t)lround(255.0 * sw / cnt);
        for (int k = 0; k < 6; k++) p.a[2 + k][o] = (uint8_t)lround(st[k] / sw);
        if (swr > 0) { const double v = 8.0 / (f->h * fp * (srho / swr)); p.a[8][o] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); }
        if (cwr > 0) { const double v = 8.0 / (f->h * fp * (crho / cwr)); p.cs[o] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); }
    }
}

/* the fibre pair term (fit_cfg fib + w_fib): points 16..128 native voxels apart along one fibre whose network phase, unwrapped along the
   fibre (nearest field voxel per native-voxel step, confidence >= 0.2, gaps of <= 2 steps bridged), changes by less than half a turn lie
   on one sheet (published Paris 4 fibres + wind96 phase: 93% one GT sheet at 0.3-1.2 mm; src/fibercheck.c): W(b) - W(a) should equal that
   path change dP (the phase is W mod 1). Loss w_fib x mean Huber(W(b) - W(a) - dP, 0.25) over the pairs, on the lattice delta (the
   knots' part is fixed per stage: base). */
typedef struct { float a[3], b[3], d, base; } fib_pair;
static fib_pair *fib_pairs(const fit_fields *f, const fit_cfg *c, size_t *np_) {
    size_t np = 0, cap = 0; fib_pair *P = nullptr; const double h = f->h;
    static const int DD[4] = {16, 32, 64, 128};
    float *q = nullptr; double *U = nullptr; size_t qcap = 0;
    for (int r = 0; r < c->fib_n; r++) {
        const float *p = c->fib + 3 * (size_t)c->fib_off[r]; const int n = c->fib_off[r + 1] - c->fib_off[r]; if (n < 2) continue;
        size_t ns = 0; double acc = 0, next = 0, last = NAN; int gap = 99;
        for (int k = 1; k < n; k++) {
            const double a[3] = {p[3 * k - 3] - c->fib_zoff, p[3 * k - 2], p[3 * k - 1]}, d[3] = {p[3 * k] - p[3 * k - 3], p[3 * k + 1] - p[3 * k - 2], p[3 * k + 2] - p[3 * k - 1]};
            const double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            for (; L > 0 && next <= acc + L; next += 1) {
                const double t = (next - acc) / L, s[3] = {a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]};
                if (ns == qcap) { qcap = qcap ? 2 * qcap : 4096; q = realloc(q, 3 * qcap * sizeof *q); U = realloc(U, qcap * sizeof *U); if (!q || !U) abort(); }
                for (int e = 0; e < 3; e++) q[3 * ns + e] = (float)s[e];
                U[ns] = NAN;
                const int iz = (int)floor(s[0] / h), iy = (int)floor(s[1] / h), ix = (int)floor(s[2] / h);
                if (iz >= 0 && iy >= 0 && ix >= 0 && iz < f->D && iy < f->H && ix < f->W) {
                    const size_t i = ((size_t)iz * f->H + iy) * f->W + ix; const double cc = (f->pc[i] - 128) / 127.0, sn = (f->ps[i] - 128) / 127.0;
                    if (f->w[i] && hypot(cc, sn) >= 0.2) {
                        const double ph = atan2(sn, cc) / (2 * M_PI);
                        U[ns] = gap <= 2 && last == last ? last + (ph - last - floor(ph - last + 0.5)) : ph;
                        if (!(gap <= 2 && last == last)) U[ns] += 1000.0 * (double)(np + ns + 1);   /* a new stretch: no pair across the break */
                        last = U[ns]; gap = 0;
                    }
                }
                if (!(U[ns] == U[ns])) gap++;
                ns++;
            }
            acc += L;
        }
        for (size_t b = 0; b < ns; b++) {
            if (!(U[b] == U[b])) continue;
            for (int di = 0; di < 4; di++) {
                if (b < (size_t)DD[di]) break;
                const size_t a = b - DD[di]; if (a % 4 || !(U[a] == U[a])) continue;
                const double dp = U[b] - U[a]; if (fabs(dp) >= 0.5) continue;   /* a turn crossed (or a break): the fibre hopped */
                if (np == cap) { cap = cap ? 2 * cap : 65536; P = realloc(P, cap * sizeof *P); if (!P) abort(); }
                fib_pair *e = &P[np++]; for (int k = 0; k < 3; k++) { e->a[k] = q[3 * a + k]; e->b[k] = q[3 * b + k]; } e->d = (float)dp; e->base = 0;
            }
        }
    }
    free(q); free(U); *np_ = np; return P;
}
/* trilinear lattice weights at a point (as lattice_at) */
static void lattice_w(int L, int lz, int ly, int lx, double z, double y, double x, size_t idx[8], double w[8]) {
    const double q[3] = {z / L, y / L, x / L}; const int dim[3] = {lz, ly, lx}; int c0[3]; double ct[3];
    for (int k = 0; k < 3; k++) {
        double a = q[k] < 0 ? 0 : q[k] > dim[k] - 1 ? dim[k] - 1 : q[k];
        c0[k] = (int)a; if (c0[k] > dim[k] - 2) c0[k] = dim[k] - 2 < 0 ? 0 : dim[k] - 2;
        ct[k] = dim[k] > 1 ? a - c0[k] : 0;
    }
    for (int c = 0; c < 8; c++) {
        const int a = c0[0] + (c >> 2 & 1), b = c0[1] + (c >> 1 & 1), e = c0[2] + (c & 1);
        w[c] = (c >> 2 & 1 ? ct[0] : 1 - ct[0]) * (c >> 1 & 1 ? ct[1] : 1 - ct[1]) * (c & 1 ? ct[2] : 1 - ct[2]);
        idx[c] = ((size_t)(a < lz ? a : lz - 1) * ly + (b < ly ? b : ly - 1)) * lx + (e < lx ? e : lx - 1);
    }
}
/* loss and its gradient (added to g) over the pairs; also the share of pairs whose rounded W difference is not 0 */
static double fib_pair_grad(const fib_pair *P, size_t n, const float *d, int L, int lz, int ly, int lx, double wt, float *g, double *off) {
    double loss = 0; size_t bad = 0; const double hub = 0.25;
    for (size_t i = 0; i < n; i++) {
        size_t ia[8], ib[8]; double wa[8], wb[8];
        lattice_w(L, lz, ly, lx, P[i].a[0], P[i].a[1], P[i].a[2], ia, wa); lattice_w(L, lz, ly, lx, P[i].b[0], P[i].b[1], P[i].b[2], ib, wb);
        double r = P[i].base - P[i].d; for (int c = 0; c < 8; c++) r += wb[c] * d[ib[c]] - wa[c] * d[ia[c]];
        bad += lround(r + P[i].d) != 0;
        const double ar = fabs(r); loss += ar < hub ? 0.5 * r * r : hub * (ar - 0.5 * hub);
        const double gr = wt / n * (ar < hub ? r : hub * (r > 0 ? 1 : -1));
        if (g) for (int c = 0; c < 8; c++) { g[ib[c]] += (float)(gr * wb[c]); g[ia[c]] -= (float)(gr * wa[c]); }
    }
    if (off) *off = n ? (double)bad / n : 0;
    return n ? wt * loss / n : 0;
}
/* ---- the CPU backend (fit_cfg cpu): nn_fit_loss2 / nn_fit_smooth / nn_adamw on host arrays (src/fit_terms.h: the same per-voxel
   terms as the CUDA kernels). Minibatches are stratified: one voxel per run of stride voxels (hash of the run and the step's salt),
   so a step touches n / stride voxels, not n hashes. Threads own contiguous voxel ranges, hence a few lattice z planes each: per
   thread a private gradient over its planes, summed into g_delta plane by plane in thread order (deterministic results). ---- */
typedef struct {
    const fit_dev *f; float *g; int nt; size_t nb, stride; double part[64][FIT_NPARTS];
    float *pg[64]; int a0[64], a1[64]; size_t plane;   /* per thread: private gradient over lattice planes [a0, a1] */
} cpu_loss;
static int cpu_plane_of(const fit_dev *f, int z) {   /* the lower lattice plane of field row z (fit_vox2's c0[0]) */
    const float a = fminf(fmaxf((z + 0.5f) * f->h / (float)f->L, 0.f), (float)(f->lz - 1));
    return FIT_IMIN((int)a, f->lz - 2 < 0 ? 0 : f->lz - 2);
}
static int cpu_plane_at(const fit_dev *f, float pz) {   /* the lower lattice plane of box-local native z (fit_U_at's c0[0]) */
    const float a = fminf(fmaxf(pz / (float)f->L, 0.f), (float)(f->lz - 1));
    return FIT_IMIN((int)a, f->lz - 2 < 0 ? 0 : f->lz - 2);
}
static void cpu_loss_setup(cpu_loss *J, const fit_dev *f, int nt) {   /* the threads' ranges and gradient buffers (per stage) */
    memset(J, 0, sizeof *J); J->f = f; J->nt = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    const size_t n = (size_t)f->D * f->H * f->W, HW = (size_t)f->H * f->W; J->stride = f->stride > 1 ? (size_t)f->stride : 1; J->nb = (n + J->stride - 1) / J->stride;
    J->plane = (size_t)f->ly * f->lx;
    for (int t = 0; t < J->nt; t++) {
        const size_t b0 = J->nb * t / J->nt, b1 = J->nb * (t + 1) / J->nt;
        if (b1 <= b0) { J->a0[t] = 0; J->a1[t] = -1; continue; }
        const size_t v0 = b0 * J->stride, v1 = b1 * J->stride < n ? b1 * J->stride : n;
        const int zl = (int)((v1 - 1) / HW);   /* the winding mod 4 term also reaches w4_d native voxels up in z */
        const int pl = f->wd ? FIT_IMIN(cpu_plane_at(f, (zl + 0.5f) * f->h + f->w4_d), f->lz - 1) : cpu_plane_of(f, zl);
        J->a0[t] = cpu_plane_of(f, (int)(v0 / HW)); J->a1[t] = FIT_IMIN((pl > cpu_plane_of(f, zl) ? pl : cpu_plane_of(f, zl)) + 1, f->lz - 1);
        J->pg[t] = malloc((size_t)(J->a1[t] - J->a0[t] + 1) * J->plane * sizeof(float));
    }
}
static void cpu_loss_free(cpu_loss *J) { for (int t = 0; t < J->nt; t++) free(J->pg[t]); }
static void cpu_loss_run(void *ctx, int t0_, int t1_, int t) {
    cpu_loss *J = ctx; const fit_dev *f = J->f; const size_t n = (size_t)f->D * f->H * f->W;
    double lp[FIT_NPARTS] = {0};
    if (J->a1[t] >= J->a0[t]) {
        float *pg = J->pg[t]; const size_t base = (size_t)J->a0[t] * J->plane, np = (size_t)(J->a1[t] - J->a0[t] + 1) * J->plane;
        memset(pg, 0, np * sizeof(float));
        const size_t b0 = J->nb * t / J->nt, b1 = J->nb * (t + 1) / J->nt; const float sc = (float)J->stride;
        for (size_t b = b0; b < b1; b++) {
            const size_t i = J->stride > 1 ? b * J->stride + fit_hash((unsigned)b ^ f->salt) % (unsigned)J->stride : b;
            if (i >= n || !f->w[i]) continue;
            size_t ic[FIT_VOX2_MAXC]; float gv[FIT_VOX2_MAXC];
            const int nc = fit_vox2(f, i, f->w[i] * (1.f / 255.f), sc, lp, ic, gv);
            for (int c = 0; c < nc; c++) pg[ic[c] - base] += gv[c];
        }
    }
    for (int k = 0; k < FIT_NPARTS; k++) J->part[t][k] = lp[k];
}
static void cpu_loss_reduce(void *ctx, int a0, int a1, int t_) {   /* lattice planes [a0, a1): the threads' gradients in thread order */
    cpu_loss *J = ctx;
    for (int a = a0; a < a1; a++) {
        float *g = J->g + (size_t)a * J->plane;
        for (int t = 0; t < J->nt; t++) {
            if (a < J->a0[t] || a > J->a1[t]) continue;
            const float *p = J->pg[t] + (size_t)(a - J->a0[t]) * J->plane;
            for (size_t k = 0; k < J->plane; k++) g[k] += p[k];
        }
    }
}
static void fit_loss2_cpu(cpu_loss *J, double *parts, float *g_delta) {
    J->g = g_delta;
    fit_zpar(J->nt, J->nt, cpu_loss_run, J);
    fit_zpar(J->f->lz, fit_nthreads(J->f->lz), cpu_loss_reduce, J);
    for (int t = 0; t < J->nt; t++) for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH) parts[k] += J->part[t][k];
}
void fit_loss2_host(const fit_dev *f, double *parts, float *g_delta) { cpu_loss J; cpu_loss_setup(&J, f, fit_nthreads(1 << 30)); fit_loss2_cpu(&J, parts, g_delta); cpu_loss_free(&J); }
typedef struct { const float *v, *wn; float *lap, *g; int lz, ly, lx; float ws, wd; double part[64]; int pass; } cpu_smooth;
static void cpu_smooth_run(void *ctx, int a0, int a1, int t) {   /* pass 0: the laplacian; pass 1: the gradient and the loss */
    cpu_smooth *S = ctx; const int ly = S->ly, lx = S->lx; const size_t pl = (size_t)ly * lx; double l = 0;
    for (int a = a0; a < a1; a++) for (int b = 0; b < ly; b++) for (int c = 0; c < lx; c++) {
        const size_t i = (size_t)a * pl + (size_t)b * lx + c;
        if (S->pass == 0) {
            const float *v = S->v; const float x = v[i]; float s = 0.f;
            if (a > 0) s += v[i - pl] - x;
            if (a < S->lz - 1) s += v[i + pl] - x;
            if (b > 0) s += v[i - lx] - x;
            if (b < ly - 1) s += v[i + lx] - x;
            if (c > 0) s += v[i - 1] - x;
            if (c < lx - 1) s += v[i + 1] - x;
            S->lap[i] = s; continue;
        }
        const float *lap = S->lap, *wn = S->wn;
#define WL(j_) (wn ? wn[j_] * lap[j_] : lap[j_])
        int deg = 0; float s = 0.f;
        if (a > 0) { s += WL(i - pl); deg++; }
        if (a < S->lz - 1) { s += WL(i + pl); deg++; }
        if (b > 0) { s += WL(i - lx); deg++; }
        if (b < ly - 1) { s += WL(i + lx); deg++; }
        if (c > 0) { s += WL(i - 1); deg++; }
        if (c < lx - 1) { s += WL(i + 1); deg++; }
        const float wli = WL(i);
#undef WL
        S->g[i] += 2.f * S->ws * (s - deg * wli) + 2.f * S->wd * S->v[i];
        l += (double)S->ws * wli * lap[i] + (double)S->wd * S->v[i] * S->v[i];
    }
    if (S->pass) S->part[t] = l;
}
static void fit_smooth_cpu(const float *delta, int lz, int ly, int lx, float ws, float wd, const float *wn, float *lap, double *parts, float *g) {
    cpu_smooth S = {delta, wn, lap, g, lz, ly, lx, ws, wd, {0}, 0}; const int nt = fit_nthreads(lz);
    fit_zpar(lz, nt, cpu_smooth_run, &S); S.pass = 1; fit_zpar(lz, nt, cpu_smooth_run, &S);
    for (int t = 0; t < nt; t++) parts[FIT_P_SMOOTH] += S.part[t];
}
typedef struct { float *p, *m, *v; const float *g; size_t n; float lr, b1, b2, eps, c1, c2; } cpu_adam;
static void cpu_adam_run(void *ctx, int t0, int t1, int t_) {
    cpu_adam *A = ctx; const size_t i0 = A->n * (size_t)t0 / 64, i1 = A->n * (size_t)t1 / 64;
    for (size_t i = i0; i < i1; i++) {
        const float gi = A->g[i], mi = A->m[i] = A->b1 * A->m[i] + (1.f - A->b1) * gi, vi = A->v[i] = A->b2 * A->v[i] + (1.f - A->b2) * gi * gi;
        A->p[i] -= A->lr * ((mi / A->c1) / (sqrtf(vi / A->c2) + A->eps));
    }
}
static void fit_adam_cpu(float *p, const float *g, float *m, float *v, size_t n, float lr, int step) {   /* nn_adamw, weight decay 0 */
    cpu_adam A = {p, m, v, g, n, lr, 0.9f, 0.999f, 1e-8f, 1.f - powf(0.9f, (float)step), 1.f - powf(0.999f, (float)step)};
    fit_zpar(64, fit_nthreads(64), cpu_adam_run, &A);
}
/* curvature-adaptive smoothness (fit_cfg bend): per lattice node the dispersion of the sheet normals around it, b = 1 - lambda_max /
   tr of the material-weighted mean n n^T over the field voxels nearest the node (0 for a plane, larger where the sheets bend or
   fold within the node's cell), and the node's smoothness weight 1 / (1 + bend 10 b) */
static float *fit_bend_weights(const fit_fields *f, int L, int lz, int ly, int lx, double bend, int verbose) {
    const size_t nl = (size_t)lz * ly * lx; double *T = calloc(nl * 7, sizeof(double)); float *wn = malloc(nl * sizeof(float));
    for (int z = 0; z < f->D; z++) for (int y = 0; y < f->H; y++) for (int x = 0; x < f->W; x++) {
        const size_t i = ((size_t)z * f->H + y) * f->W + x; if (!f->w[i]) continue;
        const int a = FIT_IMIN((int)lround((z + 0.5) * f->h / L), lz - 1), b = FIT_IMIN((int)lround((y + 0.5) * f->h / L), ly - 1), c = FIT_IMIN((int)lround((x + 0.5) * f->h / L), lx - 1);
        double *t = T + 7 * (((size_t)a * ly + b) * lx + c); const double w = f->w[i] / 255.0;
        for (int k = 0; k < 6; k++) t[k] += w * (f->t[k][i] / 255.0 - (k < 3 ? 0 : 0.5));
        t[6] += w;
    }
    double sb = 0; size_t nb = 0;
    for (size_t k = 0; k < nl; k++) {
        const double *t = T + 7 * k; wn[k] = 1.f;
        if (t[6] <= 0) continue;
        const double M[3][3] = {{t[0], t[3], t[4]}, {t[3], t[1], t[5]}, {t[4], t[5], t[2]}}, tr = t[0] + t[1] + t[2];
        if (tr <= 0) continue;
        double v[3] = {0.577, 0.577, 0.577}, lam = 0;
        for (int it = 0; it < 32; it++) {
            double u[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) u[a] += M[a][b] * v[b];
            lam = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (lam < 1e-12) break; for (int a = 0; a < 3; a++) v[a] = u[a] / lam;
        }
        const double bd = fmax(0, 1 - lam / tr); sb += bd; nb++;
        wn[k] = (float)(1 / (1 + bend * 10 * bd));
    }
    if (verbose) fprintf(stderr, "fit: curvature-adaptive smoothness at lattice %d: mean normal dispersion %.4f over %zu nodes\n", L, nb ? sb / nb : 0, nb);
    free(T);
    return wn;
}

/* buffers of the device (GPU) or of the host (CPU backend) */
static void *fb_alloc(int cpu, size_t n) { return cpu ? calloc(1, n ? n : 4) : nn_malloc(n ? n : 4); }
static void fb_free(int cpu, void *p) { if (cpu) free(p); else nn_free(p); }
static void fb_put(int cpu, void *d, const void *s, size_t n) { if (cpu) memcpy(d, s, n); else nn_h2d(d, s, n); }
static void fb_get(int cpu, void *d, const void *s, size_t n) { if (cpu) memcpy(d, s, n); else nn_d2h(d, s, n); }
static void fb_zero(int cpu, void *d, size_t n) { if (cpu) memset(d, 0, n); else nn_zero(d, n); }

static fit_model *fit_run_dir(const fit_fields *f, const fit_cfg *c) {
    const int cpu = c->cpu;
    if (cpu && !c->analytic) { fprintf(stderr, "fit: the CPU backend fits the analytic terms only (--analytic 1)\n"); return nullptr; }
    g_fit_threads = c->threads;
    fit_model *m = calloc(1, sizeof *m);
    m->D = f->D; m->H = f->H; m->W = f->W; m->h = f->h;
    const int na = (int)(f->D * f->h) + 1;
    m->cy = malloc(na * sizeof(float)); m->cx = malloc(na * sizeof(float));
    memcpy(m->cy, f->cy, na * sizeof(float)); memcpy(m->cx, f->cx, na * sizeof(float));
    if (c->init) {   /* the given model's direction and knots (sampled at this box's knot positions) */
        m->dir = c->init->dir; m->dir_vote = 1; m->knot_dz = c->knot_dz;
        m->nk = (int)ceil(f->D * f->h / c->knot_dz) + 1; if (m->nk < 2) m->nk = 2;
        m->pitch = calloc(m->nk, sizeof(float)); m->offset = calloc(m->nk, sizeof(float));
        for (int k = 0; k < m->nk; k++) {
            const double zg = (double)k * c->knot_dz + c->init_off[0];
            m->pitch[k] = (float)knot_lerp(c->init->pitch, c->init->nk, c->init->knot_dz, zg);
            m->offset[k] = (float)knot_lerp(c->init->offset, c->init->nk, c->init->knot_dz, zg);
        }
    } else { const double tp = nn_wall_seconds(); if (fit_init_prior(f, c, m)) { fit_free(m); return nullptr; } if (getenv("UFSM_FIT_TIMES")) fprintf(stderr, "fit times: prior %.2fs\n", nn_wall_seconds() - tp); }
    const size_t n = (size_t)f->D * f->H * f->W;
    size_t ndata = 0; for (size_t i = 0; i < n; i++) ndata += f->w[i] != 0;
    const uint8_t *dpc = f->pc, *dps = f->ps, *dt[6], *dsp = f->sp, *dw = f->w; uint8_t *dfield = nullptr;   /* CPU: the host fields as they are */
    for (int k = 0; k < 6; k++) dt[k] = f->t[k];
    fit_stream *fst = cpu ? nullptr : c->stream;   /* streamed fields: already on the device, the pooled stages made while loading */
    if (fst && fst->n != n) { fprintf(stderr, "fit: the streamed fields do not match the box\n"); fit_free(m); return nullptr; }
    if (fst) {
        dfield = fst->dfield;
        dpc = dfield; dps = dfield + n; dsp = dfield + 8 * n; dw = dfield + 9 * n; for (int k = 0; k < 6; k++) dt[k] = dfield + (2 + k) * n;
    } else if (!cpu) {
        /* the fields on the device; when they do not fit (or UFSM_FIT_MANAGED=1), in CUDA managed memory: pages migrate on demand, so boxes
           larger than the GPU (e.g. 9.6 um cells over a full cross-section) still fit, slower */
        const int force_managed = ufsm_env_on("UFSM_FIT_MANAGED");
        dfield = force_managed ? nullptr : nn_malloc_try(10 * n);
        if (!dfield) {
            dfield = nn_managed_alloc(10 * n);
            if (dfield && c->verbose) fprintf(stderr, "fit: fields in managed memory (%.1f GB, %s)\n", 10.0 * n / 1e9, force_managed ? "UFSM_FIT_MANAGED" : "over the free device memory");
        }
        if (!dfield) { fprintf(stderr, "fit: no device memory for the fields (%.1f GB: 10 B per cell of %dx%dx%d); use a smaller box or --slab\n", 10.0 * n / 1e9, f->D, f->H, f->W); fit_free(m); return nullptr; }
        nn_h2d(dfield, f->pc, n); nn_h2d(dfield + n, f->ps, n); nn_h2d(dfield + 8 * n, f->sp, n); nn_h2d(dfield + 9 * n, f->w, n);
        for (int k = 0; k < 6; k++) nn_h2d(dfield + (2 + k) * n, f->t[k], n);
        dpc = dfield; dps = dfield + n; dsp = dfield + 8 * n; dw = dfield + 9 * n; for (int k = 0; k < 6; k++) dt[k] = dfield + (2 + k) * n;
    }
    float *dcy = fb_alloc(cpu, na * 4), *dcx = fb_alloc(cpu, na * 4);
    fb_put(cpu, dcy, f->cy, na * 4); fb_put(cpu, dcx, f->cx, na * 4);
    const int nk = m->nk;
    float *dpi = fb_alloc(cpu, nk * 4), *dof = fb_alloc(cpu, nk * 4), *gpi = fb_alloc(cpu, nk * 4), *gof = fb_alloc(cpu, nk * 4);
    float *mpi = fb_alloc(cpu, nk * 4), *vpi = fb_alloc(cpu, nk * 4), *mof = fb_alloc(cpu, nk * 4), *vof = fb_alloc(cpu, nk * 4);
    fb_put(cpu, dpi, m->pitch, nk * 4); fb_put(cpu, dof, m->offset, nk * 4);
    fb_zero(cpu, mpi, nk * 4); fb_zero(cpu, vpi, nk * 4); fb_zero(cpu, mof, nk * 4); fb_zero(cpu, vof, nk * 4);
    double *dparts = fb_alloc(cpu, FIT_NPARTS * sizeof(double));
    uint8_t *dwd = nullptr, *drec = nullptr;   /* the winding mod 4 planes (dz dy dx dq back to back) and the recto plane, full resolution */
    if (f->wd[0] && f->wd[1] && f->wd[2] && f->wd[3] && f->w4_d > 0 && c->w_wind4 > 0 && c->analytic) {
        dwd = fb_alloc(cpu, 4 * n); for (int k = 0; k < 4; k++) fb_put(cpu, dwd + k * n, f->wd[k], n);
        if (c->verbose) fprintf(stderr, "fit: winding mod 4 term (offset %d native voxels, weight %g)\n", f->w4_d, c->w_wind4);
    }
    if (f->recto && c->w_recto > 0) {
        drec = fb_alloc(cpu, n); fb_put(cpu, drec, f->recto, n);
        if (c->verbose) fprintf(stderr, "fit: recto anchors (weight %g)\n", c->w_recto);
    }
    uint8_t *hctm = nullptr, *dctm = nullptr, *dcs = nullptr;   /* the CT parity (host for the flood, device / host for the terms), the CT spacing */
    if (f->ct && c->w_ct > 0) {
        const int pr = (int)lround(c->ct_radius_um / (f->h * (c->um > 0 ? c->um : 2.4)));
        hctm = malloc(n); fit_ct_parity(f->ct, f->D, f->H, f->W, pr, hctm);
        if (cpu) dctm = hctm; else { dctm = nn_malloc(n); nn_h2d(dctm, hctm, n); }
        if (c->verbose) { size_t np = 0, nv = 0; for (size_t i = 0; i < n; i++) if (f->w[i] && hctm[i] != 128) { np += hctm[i] > 128; nv += hctm[i] < 128; }
                          fprintf(stderr, "fit: CT parity (weight %g, (1 - conf)^%g >= %g, papyrus phase %.3f, radius %d cells): %zu papyrus / %zu void cells\n", c->w_ct, c->ct_pow, c->ct_floor, c->ct_phi, pr, np, nv); }
        if (!cpu && !c->ct_flood) { free(hctm); hctm = nullptr; }   /* on the device only (the flood reads it on the host) */
    }
    if (f->cs && c->w_cspacing > 0) {
        dcs = fb_alloc(cpu, n); fb_put(cpu, dcs, f->cs, n);
        if (c->verbose) fprintf(stderr, "fit: CT sheet spacing term (weight %g, tolerance %g)\n", c->w_cspacing, c->cs_tol);
    }
    const double norm = ndata ? 1.0 / ndata : 1;
    int kstep = 0;
    size_t nfp = 0; fib_pair *fpairs = c->fib && c->fib_n > 0 && c->w_fib > 0 ? fib_pairs(f, c, &nfp) : nullptr;
    float *fph = nullptr, *fpg = nullptr;   /* host copies of delta and its gradient (GPU backend) */
    if (c->verbose && fpairs) fprintf(stderr, "fit: fibre pair term: %zu pairs (weight %g)\n", nfp, (double)c->w_fib);
    for (int st = 0; st < c->nstage; st++) {
        const double tst = nn_wall_seconds();
        const int L = c->lattice[st], lz = (int)ceil(f->D * f->h / L) + 1, ly = (int)ceil(f->H * f->h / L) + 1, lx = (int)ceil(f->W * f->h / L) + 1;
        const size_t nl = (size_t)lz * ly * lx;
        float *hd = prolong(m, L, lz, ly, lx, c);
        free(m->delta); m->delta = hd; m->L = L; m->lz = lz; m->ly = ly; m->lx = lx;
        if (nfp) {   /* the knots' part of each pair's W difference (fixed during the stage) */
            for (size_t i = 0; i < nfp; i++) { double ta, tb; const double Wa = fit_base(m, fpairs[i].a[0], fpairs[i].a[1], fpairs[i].a[2], &ta) - m->dir * ta / (2 * M_PI), Wb = fit_base(m, fpairs[i].b[0], fpairs[i].b[1], fpairs[i].b[2], &tb) - m->dir * tb / (2 * M_PI);
                double dth = (tb - ta) / (2 * M_PI); dth -= floor(dth + 0.5); fpairs[i].base = (float)((Wb + m->dir * tb / (2 * M_PI)) - (Wa + m->dir * ta / (2 * M_PI)) - m->dir * dth); }
            if (!cpu) { free(fph); free(fpg); fph = malloc(nl * 4); fpg = malloc(nl * 4); }
        }
        if (fst && fst->drop_pooled && st == c->phase_stage)   /* the pooled stages are done (one direction: no second run) */
            for (int q = 0; q < fst->nfp; q++) if (fst->dpool[q]) { nn_free(fst->dpool[q]); fst->dpool[q] = nullptr; }
        if (st == c->phase_stage && c->w_phase > 0) {   /* the current knots too: W reads them */
            fb_get(cpu, m->pitch, dpi, nk * 4); fb_get(cpu, m->offset, dof, nk * 4);
            const double umc = c->uw_min_conf > 0 ? c->uw_min_conf : 0.3;
            if (c->unwrap_voxels > 0) {
                const fit_fields *fu = f; fit_fields fm; uint8_t *mpc = nullptr, *mps = nullptr; double um2 = umc;
                if (hctm && c->ct_flood) {   /* voxels below the network's floor flood with the CT's phase, after every network voxel */
                    const double tq = nn_wall_seconds(), qmin = 0.3; size_t nadd = 0;
                    mpc = malloc(n); mps = malloc(n); fit_ct_phasor(m, f, hctm, c->ct_phi, mpc, mps);
                    for (size_t i = 0; i < n; i++) {
                        const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0;
                        if (sqrt(mc * mc + ms * ms) >= umc) { mpc[i] = f->pc[i]; mps[i] = f->ps[i]; continue; }
                        const double cc = (mpc[i] - 128) / 127.0, cs_ = (mps[i] - 128) / 127.0, q = sqrt(cc * cc + cs_ * cs_);
                        if (q < qmin) { mpc[i] = mps[i] = 128; continue; }
                        const double sc = 0.999 * umc;   /* below every network voxel: confidence umc x q (q in qmin .. 1) */
                        mpc[i] = (uint8_t)lround(128 + 127 * cc * sc); mps[i] = (uint8_t)lround(128 + 127 * cs_ * sc); nadd++;
                    }
                    fm = *f; fm.pc = mpc; fm.ps = mps; fu = &fm; um2 = umc * qmin * 0.95;
                    if (c->verbose) fprintf(stderr, "fit: CT phase joins the voxel unwrapping on %zu low-confidence voxels (%.1fs)\n", nadd, nn_wall_seconds() - tq);
                }
                const size_t rch = fit_unwrap_voxels(m, fu, um2, c); if (c->verbose) fprintf(stderr, "fit: voxel unwrapping reached %zu field voxels\n", rch);
                free(mpc); free(mps);
            }
            else if (c->unwrap_voxels == 0) fit_unwrap_init(m, f, umc);
        }
        float *dd = fb_alloc(cpu, nl * 4), *gd = fb_alloc(cpu, nl * 4), *md = fb_alloc(cpu, nl * 4), *vd = fb_alloc(cpu, nl * 4), *lap = fb_alloc(cpu, nl * 4);
        fb_put(cpu, dd, hd, nl * 4); fb_zero(cpu, md, nl * 4); fb_zero(cpu, vd, nl * 4);
        float *dwn = nullptr;   /* curvature-adaptive smoothness weights per lattice node */
        if (c->bend > 0) { float *hw = fit_bend_weights(f, L, lz, ly, lx, c->bend, c->verbose); dwn = fb_alloc(cpu, nl * 4); fb_put(cpu, dwn, hw, nl * 4); free(hw); }
        int fp = 1; while (st < c->phase_stage && 2 * fp * f->h <= L / 4.0) fp *= 2;   /* gradient-only stages: pooled fields (~4 field voxels per lattice cell) */
        fit_dev fd = {f->D, f->H, f->W, (float)f->h, dpc, dps, {dt[0], dt[1], dt[2], dt[3], dt[4], dt[5]}, dsp, dw, dcy, dcx, na, m->dir, nk, m->knot_dz, dpi, dof, L, lz, ly, lx, dd,
                      0, (float)(c->w_normal * norm), (float)(c->w_density * norm), (float)(c->w_nondeg * norm), c->core, 0, 0u, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, nullptr, nullptr, 0.f, 0.f, nullptr, 0.f, nullptr, 0.f, 1.f, 0.f, 0.f, nullptr, 0.f, 0.25f};
        fd.phase_robust = c->phase_robust; fd.conf_pow = c->conf_pow > 0 ? c->conf_pow : 1.f;
        fd.w_pgrad = (float)(c->w_pgrad * norm); fd.pgrad_delta = c->pgrad_delta; fd.pg = nullptr;
        fd.wd = fp == 1 ? dwd : nullptr; fd.w4_d = (float)f->w4_d; fd.recto = fp == 1 ? drec : nullptr;   /* (pooled stages: neither; their ramp is 0 there anyway) */
        fd.ctm = fp == 1 ? dctm : nullptr; fd.ct_pow = c->ct_pow; fd.ct_phi = c->ct_phi; fd.ct_floor = c->ct_floor;
        fd.cs = fp == 1 ? dcs : nullptr; fd.w_cs = (float)(c->w_cspacing * norm); fd.cs_tol = c->cs_tol;
        uint8_t *pcs = nullptr;   /* the pooled CT spacing (device or host) */
        float *pgd = nullptr;   /* pooled phase gradients (device or host) */
        uint8_t *pdev = nullptr, *phost[10] = {0};
        if (fp > 1) {
            int sq = -1; if (fst) for (int q = 0; q < fst->nfp; q++) if (fst->fp[q] == fp && fst->dpool[q]) sq = q;
            if (fst && sq < 0) { fprintf(stderr, "fit: no streamed fields pooled by %d\n", fp); fit_free(m); return nullptr; }   /* (fit_stream_fps: never) */
            fit_pooled pp = sq < 0 ? fit_pool(f, fp, c->w_pgrad > 0) : (fit_pooled){fst->pD[sq], fst->pH[sq], fst->pW[sq], {0}, nullptr, nullptr};
            const size_t np = (size_t)pp.D * pp.H * pp.W;
            size_t pdata = 0;
            if (sq >= 0) { unsigned long long cnt[8]; nn_d2h(cnt, fst->dcnt, sizeof cnt); pdata = (size_t)cnt[sq]; }   /* (the device planes: fit_stream_free) */
            else for (size_t i = 0; i < np; i++) pdata += pp.a[9][i] != 0;
            const uint8_t *pa[10];
            if (sq >= 0) for (int k = 0; k < 10; k++) pa[k] = fst->dpool[sq] + (size_t)k * np;
            else if (cpu) { for (int k = 0; k < 10; k++) pa[k] = phost[k] = pp.a[k]; }
            else {
                pdev = nn_malloc(10 * np);
                if (!pdev) { fprintf(stderr, "fit: out of GPU memory for the fields pooled by %d (%.1f GB): crop the fields in y / x (the papyrus box) or use a larger GPU\n", fp, 10.0 * np / 1e9); fit_free(m); return nullptr; }
                for (int k = 0; k < 10; k++) { nn_h2d(pdev + k * np, pp.a[k], np); pa[k] = pdev + k * np; if (sq < 0) free(pp.a[k]); }
            }
            fd.D = pp.D; fd.H = pp.H; fd.W = pp.W; fd.h = (float)(f->h * fp);
            fd.pc = pa[0]; fd.ps = pa[1]; for (int k = 0; k < 6; k++) fd.t[k] = pa[2 + k]; fd.sp = pa[8]; fd.w = pa[9];
            const double pn = pdata ? 1.0 / pdata : 1;
            fd.w_normal = (float)(c->w_normal * pn); fd.w_density = (float)(c->w_density * pn); fd.w_nondeg = (float)(c->w_nondeg * pn); fd.w_pgrad = (float)(c->w_pgrad * pn);
            if (pp.pg) { pgd = fb_alloc(cpu, 4 * np * sizeof(float)); fb_put(cpu, pgd, pp.pg, 4 * np * sizeof(float)); free(pp.pg); fd.pg = pgd; }
            if (pp.cs && dcs) { pcs = fb_alloc(cpu, np); fb_put(cpu, pcs, pp.cs, np); fd.cs = pcs; fd.w_cs = (float)(c->w_cspacing * pn); }
            free(pp.cs);
            if (c->verbose) fprintf(stderr, "fit stage %d: fields pooled by %d (%dx%dx%d)\n", st, fp, pp.D, pp.H, pp.W);
        }
        const float ws = (float)(c->w_smooth / nl), wdec = (float)(c->w_decay / nl);
        const size_t nvox = (size_t)fd.D * fd.H * fd.W;
        fd.stride = c->batch > 0 && nvox > (size_t)c->batch ? (int)(nvox / c->batch) : 1;
        cpu_loss cl; if (cpu) cpu_loss_setup(&cl, &fd, fit_nthreads(1 << 30));
        for (int it = 0; it < c->steps[st]; it++) {
            const double ramp = st < c->phase_stage ? 0 : st == c->phase_stage ? c->phase_ramp + (1 - c->phase_ramp) * (double)it / (c->steps[st] > 1 ? c->steps[st] - 1 : 1) : 1;
            fd.w_phase = (float)(c->w_phase * ramp * norm);
            fd.w_w4 = (float)(c->w_wind4 * ramp * norm); fd.w_recto = (float)(c->w_recto * ramp * norm); fd.w_ct = (float)(c->w_ct * ramp * norm);
            fd.packed_sp = c->packed_um > 0 ? (float)(c->packed_um / (c->um > 0 ? c->um : 2.4) * 8.0 / f->h) : 0.f; fd.packed_floor = 0.1f;
            fb_zero(cpu, gd, nl * 4); fb_zero(cpu, gpi, nk * 4); fb_zero(cpu, gof, nk * 4); fb_zero(cpu, dparts, FIT_NPARTS * sizeof(double));
            fd.salt = (unsigned)(kstep * 2654435761u + 12345u);
            double fpl = 0, fpo = 0;
            if (cpu) {
                fit_loss2_cpu(&cl, dparts, gd);
                fit_smooth_cpu(dd, lz, ly, lx, ws, wdec, dwn, lap, dparts, gd);
                if (nfp) fpl = fib_pair_grad(fpairs, nfp, dd, L, lz, ly, lx, c->w_fib, gd, &fpo);
                fit_adam_cpu(dd, gd, md, vd, nl, c->lr, it + 1);
                ++kstep;   /* the knots are fixed (analytic terms) */
            } else {
                if (c->analytic) nn_fit_loss2(&fd, dparts, gd);   /* random voxel subsets of ~batch per step */
                else nn_fit_loss(&fd, dparts, gd, gpi, gof);
                nn_fit_smooth(dd, lz, ly, lx, ws, wdec, dwn, lap, dparts, gd);
                if (nfp) { fb_get(cpu, fph, dd, nl * 4); fb_get(cpu, fpg, gd, nl * 4); fpl = fib_pair_grad(fpairs, nfp, fph, L, lz, ly, lx, c->w_fib, fpg, &fpo); fb_put(cpu, gd, fpg, nl * 4); }
                nn_adamw(dd, gd, md, vd, nl, c->lr, 0.9f, 0.999f, 1e-8f, 0.f, it + 1);
                ++kstep;
                nn_adamw(dof, gof, mof, vof, nk, c->lr_knot, 0.9f, 0.999f, 1e-8f, 0.f, kstep);
                nn_adamw(dpi, gpi, mpi, vpi, nk, c->lr_knot * 20.f, 0.9f, 0.999f, 1e-8f, 0.f, kstep);
            }
            if (c->verbose && (it == c->steps[st] - 1 || (it + 1) % 100 == 0)) {
                double p[FIT_NPARTS]; if (!cpu) nn_sync(); fb_get(cpu, p, dparts, sizeof p);
                fprintf(stderr, "fit stage %d (lattice %d, %zu nodes) step %d: loss %.5f phase %.5f normal %.5f density %.5f nondeg %.5f pgrad %.5f smooth %.6f",
                        st, L, nl, it + 1, p[0] + p[6], p[1], p[2], p[3], p[4], p[5], p[6]);
                if (dwd) fprintf(stderr, " wind4 %.5f", p[FIT_P_WIND4]);
                if (drec) fprintf(stderr, " recto %.5f", p[FIT_P_RECTO]);
                if (dctm) fprintf(stderr, " ct %.5f", p[FIT_P_CT]);
                if (dcs) fprintf(stderr, " cspace %.5f", p[FIT_P_CSPACE]);
                if (nfp) fprintf(stderr, " fibre %.5f (pairs off a turn %.3f)", fpl, fpo);
                fprintf(stderr, "\n");
            }
        }
        if (cpu) cpu_loss_free(&cl); else nn_sync();
        fb_get(cpu, hd, dd, nl * 4);
        fb_get(cpu, m->loss, dparts, FIT_NPARTS * sizeof(double)); m->loss[FIT_P_TOTAL] += m->loss[FIT_P_SMOOTH];
        if (c->verbose) fprintf(stderr, "fit stage %d: %.1fs\n", st, nn_wall_seconds() - tst);
        fb_free(cpu, dd); fb_free(cpu, gd); fb_free(cpu, md); fb_free(cpu, vd); fb_free(cpu, lap); if (dwn) fb_free(cpu, dwn); if (pdev) nn_free(pdev); if (pgd) fb_free(cpu, pgd); if (pcs) fb_free(cpu, pcs);
        for (int k = 0; k < 10; k++) free(phost[k]);
    }
    fb_get(cpu, m->pitch, dpi, nk * 4); fb_get(cpu, m->offset, dof, nk * 4);
    if (dfield && !fst) nn_free(dfield);   /* (streamed fields: the caller frees them) */
    fb_free(cpu, dcy); fb_free(cpu, dcx);
    fb_free(cpu, dpi); fb_free(cpu, dof); fb_free(cpu, gpi); fb_free(cpu, gof); fb_free(cpu, mpi); fb_free(cpu, vpi); fb_free(cpu, mof); fb_free(cpu, vof); fb_free(cpu, dparts);
    if (dwd) fb_free(cpu, dwd);
    if (drec) fb_free(cpu, drec);
    if (dctm && !cpu) nn_free(dctm);
    free(fpairs); free(fph); free(fpg);
    free(hctm); if (dcs) fb_free(cpu, dcs);
    return m;
}

/* the direction from the prior's ring vote; when it is ambiguous (|median winding per ring| < 0.5 or too few rings) both
   directions are fitted and the lower final loss wins (a wrong direction cannot fit the phase around the circle) */
/* --phase-stretch S (fit_cfg phase_stretch): the phase confidence times exp(-log2(s)^2 / 2 S^2), s the phase stretch (uwr_stretch:
   |grad phase| x spacing over a (2r+1)^3 box; 1 counts every sheet, < 1 skips sheets, > 1 adds some). The network phase skips ~40% of
   the sheets in packed Paris 4 (fit-compare: -1.2..-1.9 turns per 0.5 mm radial segment of ~2.75 against +0.0 by density), and the
   error lies inside the flooded pieces where no integer vote can reach it; with the phase gated off there the density term sets the
   turn count, and the wrapped phase term elsewhere costs nothing for the integer turns added in between. */
typedef struct { const fit_fields *f; uint8_t *pc, *ps; double S; int r; } pst_job;
static void pst_rows(void *a, int z0, int z1, int t) {
    pst_job *J = a; const fit_fields *f = J->f; (void)t;
    for (int z = z0; z < z1; z++) for (int y = 0; y < f->H; y++) for (int x = 0; x < f->W; x++) {
        const size_t i = ((size_t)z * f->H + y) * f->W + x; J->pc[i] = f->pc[i]; J->ps[i] = f->ps[i];
        if (f->pc[i] == 128 && f->ps[i] == 128) continue;
        const double l = log2(fmax(uwr_stretch(f, z, y, x, J->r), 1e-3)), g = exp(-l * l / (2 * J->S * J->S));
        J->pc[i] = (uint8_t)lround(128 + (f->pc[i] - 128) * g); J->ps[i] = (uint8_t)lround(128 + (f->ps[i] - 128) * g);
    }
}
fit_model *fit_run(const fit_fields *f, const fit_cfg *c) {
    if (c->phase_stretch > 0 && c->phase_stage < c->nstage) {
        const size_t n = (size_t)f->D * f->H * f->W; uint8_t *pc = malloc(n), *ps = malloc(n);
        if (!pc || !ps) { free(pc); free(ps); fprintf(stderr, "fit: out of memory for --phase-stretch\n"); return nullptr; }
        pst_job J = {f, pc, ps, c->phase_stretch, c->phase_stretch_r};
        fit_zpar(f->D, fit_nthreads(f->D), pst_rows, &J);
        if (c->verbose) {
            double s0 = 0, s1 = 0;
            for (size_t i = 0; i < n; i++) { s0 += hypot(f->pc[i] - 128, f->ps[i] - 128); s1 += hypot(pc[i] - 128, ps[i] - 128); }
            fprintf(stderr, "fit: --phase-stretch %.3g: phase confidence kept %.3f\n", c->phase_stretch, s0 > 0 ? s1 / s0 : 0);
        }
        fit_fields g = *f; g.pc = pc; g.ps = ps; fit_cfg c2 = *c; c2.phase_stretch = 0;
        fit_model *m = fit_run(&g, &c2); free(pc); free(ps); return m;
    }
    if (c->w_recto > 0 && c->phase_stage < c->nstage && !f->recto) { fprintf(stderr, "fit: --w-recto needs an r plane (r.raw or an r zarr array) on the fit-field grid\n"); return nullptr; }
    if (c->w_ct > 0 && c->phase_stage < c->nstage && !f->ct) { fprintf(stderr, "fit: --w-ct needs a ct plane (ufsm ct-fields, predict --fit-ct)\n"); return nullptr; }
    if (c->w_cspacing > 0 && !f->cs && c->verbose) fprintf(stderr, "fit: --w-cspacing: the fields carry no cs plane (term off)\n");
    fit_model *m = fit_run_dir(f, c);
    if (!m || c->dir || m->dir_vote >= 0.5) return m;
    fit_cfg c2 = *c; c2.dir = -m->dir;
    fit_model *m2 = fit_run_dir(f, &c2);
    if (c->verbose && m2) fprintf(stderr, "fit: ambiguous direction vote %.3f: dir %+d loss %.5f, dir %+d loss %.5f\n", m->dir_vote, m->dir, m->loss[0], m2->dir, m2->loss[0]);
    if (m2 && m2->loss[0] < m->loss[0]) { fit_free(m); return m2; }
    fit_free(m2); return m;
}

void fit_free(fit_model *m) {
    if (!m) return;
    free(m->pitch); free(m->offset); free(m->delta); free(m->cy); free(m->cx); free(m);
}

int fit_save(const fit_model *m, const char *dir) {
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", dir); if (system(p)) return -1;
    snprintf(p, sizeof p, "%s/model.json", dir);
    FILE *f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "{\"version\": 1, \"dir\": %d, \"knot_dz\": %d, \"L\": %d, \"lattice\": [%d, %d, %d], \"shape\": [%d, %d, %d], \"h\": %.9g, \"um\": %.9g, "
               "\"origin\": [%lld, %lld, %lld], \"loss\": %.9g, \"pitch\": [", m->dir, m->knot_dz, m->L, m->lz, m->ly, m->lx, m->D, m->H, m->W, m->h, m->um > 0 ? m->um : 2.4,
            m->origin[0], m->origin[1], m->origin[2], m->loss[0]);
    for (int k = 0; k < m->nk; k++) fprintf(f, "%s%.9g", k ? ", " : "", m->pitch[k]);
    fprintf(f, "], \"offset\": [");
    for (int k = 0; k < m->nk; k++) fprintf(f, "%s%.9g", k ? ", " : "", m->offset[k]);
    fprintf(f, "]}\n");
    if (fclose(f)) return -1;
    const size_t nl = (size_t)m->lz * m->ly * m->lx, na = (size_t)(m->D * m->h) + 1;
    snprintf(p, sizeof p, "%s/delta.f32", dir); f = fopen(p, "wb");
    if (!f || fwrite(m->delta, 4, nl, f) != nl) { if (f) fclose(f); return -1; } fclose(f);
    snprintf(p, sizeof p, "%s/axis.f32", dir); f = fopen(p, "wb");
    if (!f || fwrite(m->cy, 4, na, f) != na || fwrite(m->cx, 4, na, f) != na) { if (f) fclose(f); return -1; }
    return fclose(f) ? -1 : 0;
}

fit_model *fit_load(const char *dir) {
    char p[1400]; snprintf(p, sizeof p, "%s/model.json", dir);
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(n + 1); if (fread(t, 1, n, f) != n) { fclose(f); free(t); return nullptr; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); if (!j) return nullptr;
    fit_model *m = calloc(1, sizeof *m);
    m->dir = (int)json_num(json_get(j, "dir"), 1); m->knot_dz = (int)json_num(json_get(j, "knot_dz"), 1024); m->L = (int)json_num(json_get(j, "L"), 64);
    m->lz = (int)json_num(json_at(json_get(j, "lattice"), 0), 0); m->ly = (int)json_num(json_at(json_get(j, "lattice"), 1), 0); m->lx = (int)json_num(json_at(json_get(j, "lattice"), 2), 0);
    m->D = (int)json_num(json_at(json_get(j, "shape"), 0), 0); m->H = (int)json_num(json_at(json_get(j, "shape"), 1), 0); m->W = (int)json_num(json_at(json_get(j, "shape"), 2), 0);
    m->h = json_num(json_get(j, "h"), 8); m->loss[0] = json_num(json_get(j, "loss"), 0); m->um = json_num(json_get(j, "um"), 2.4);
    for (int d = 0; d < 3; d++) m->origin[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0);
    const json *pj = json_get(j, "pitch"), *oj = json_get(j, "offset");
    m->nk = pj ? (int)pj->n : 0;
    m->pitch = malloc(m->nk * sizeof(float)); m->offset = malloc(m->nk * sizeof(float));
    for (int k = 0; k < m->nk; k++) { m->pitch[k] = (float)json_num(json_at(pj, k), 60); m->offset[k] = (float)json_num(json_at(oj, k), 0); }
    json_free(j);
    const size_t nl = (size_t)m->lz * m->ly * m->lx, na = (size_t)(m->D * m->h) + 1;
    m->delta = malloc(nl * 4); m->cy = malloc(na * 4); m->cx = malloc(na * 4);
    snprintf(p, sizeof p, "%s/delta.f32", dir); f = fopen(p, "rb");
    if (!f || fread(m->delta, 4, nl, f) != nl) { if (f) fclose(f); fit_free(m); return nullptr; } fclose(f);
    snprintf(p, sizeof p, "%s/axis.f32", dir); f = fopen(p, "rb");
    if (!f || fread(m->cy, 4, na, f) != na || fread(m->cx, 4, na, f) != na) { if (f) fclose(f); fit_free(m); return nullptr; }
    fclose(f);
    return m;
}

/* the root of W = k along the ray (theta) from the axis nearest r_hint, W evaluated by ev; the axis, box and pitch of m */
typedef double (*ray_ev)(const void *ctx, double z, double y, double x, double r, double th);
enum { RR_SEG = 16, RR_SUB = 8, RR_DEPTH = 2 };
/* a sign change of W - k over [a, b]: a crossing interval when W moves by less than half a turn across it (a larger jump is the
   theta cut of a nearby ray), else the interval in RR_SUB parts, down to RR_DEPTH levels (dr / 64): where sheets are packed tighter
   than the prior pitch W legitimately moves > 0.5 turn per pitch / 4 step, and those crossings were all dropped (a fit to Paris4
   network fields lost 37-52% of its wrap vertices so) */
static int rr_cross(ray_ev evr, const void *ctx, double z, double cy, double cx, double sy, double sx, double thw, double k,
                    double a, double fa, double b, double fb, int depth, double seg[][4], int ns) {
    if (fabs(fa - fb) < 0.5) { if (ns < RR_SEG) { seg[ns][0] = a; seg[ns][1] = b; seg[ns][2] = fa; seg[ns][3] = fb; ns++; } return ns; }
    if (depth >= RR_DEPTH) return ns;
    double r0 = a, f0 = fa;
    for (int q = 1; q <= RR_SUB; q++) {
        const double r1 = q == RR_SUB ? b : a + (b - a) * q / RR_SUB, f1 = q == RR_SUB ? fb : evr(ctx, z, cy + r1 * sy, cx + r1 * sx, r1, thw) - k;
        if ((f0 < 0) != (f1 < 0)) ns = rr_cross(evr, ctx, z, cy, cx, sy, sx, thw, k, r0, f0, r1, f1, depth + 1, seg, ns);
        r0 = r1; f0 = f1;
    }
    return ns;
}
/* ua, ub (ufsm wraps --fields): every radius whose point can be a kept vertex lies in [ua, ub] (ua > ub: none). A root outside it is
   dropped by the caller, so the search returns -1 as soon as every root it can still find is outside: the ring's crossing intervals
   all outside (no refinement), or each side's front past its bound (or stopped). The same roots where they are kept. */
typedef void (*ray_prep)(const void *ctx, double thw);   /* (optional: per-ray constants of the evaluator, given the ray angle) */
static inline int ray_root_b(ray_ev evr, ray_prep prep, const void *ctx, const fit_model *m, double z, double theta, double k, double r_hint, double ua, double ub, double *r);
static int ray_root(ray_ev evr, ray_prep prep, const void *ctx, const fit_model *m, double z, double theta, double k, double r_hint, double *r) { return ray_root_b(evr, prep, ctx, m, z, theta, k, r_hint, -1e300, 1e300, r); }
static inline __attribute__((always_inline)) int ray_root_b(ray_ev evr, ray_prep prep, const void *ctx, const fit_model *m, double z, double theta, double k, double r_hint, double ua, double ub, double *r) {   /* (inlined: a constant evaluator inlines too) */
    const int na = (int)(m->D * m->h) + 1;
    const double cy = axis_lerp(m->cy, na, z), cx = axis_lerp(m->cx, na, z), sy = sin(theta), sx = cos(theta), thw = atan2(sy, sx);
    if (prep) prep(ctx, thw);
    const double Y = m->H * m->h, X = m->W * m->h;
    double rlo = 0, rhi = 1e9;   /* the part of the ray inside the box */
    const double c0[2] = {cy, cx}, dv[2] = {sy, sx}, ext[2] = {Y, X};
    for (int d = 0; d < 2; d++) {
        if (fabs(dv[d]) < 1e-12) { if (c0[d] < 0 || c0[d] > ext[d]) return -1; continue; }
        double a = (0 - c0[d]) / dv[d], b = (ext[d] - c0[d]) / dv[d]; if (a > b) { double t = a; a = b; b = t; }
        if (a > rlo) rlo = a;
        if (b < rhi) rhi = b;
    }
    if (rhi <= rlo || ua > ub) return -1;
    const double pit = knot_lerp(m->pitch, m->nk, m->knot_dz, z), dr = pit / 4;
    /* intervals of dr stepping outward from the hint on both sides; the first ring of intervals with a crossing (|jump| < half a
       turn: not the theta cut of a nearby ray) holds the crossing nearest the hint (to within dr). W grows ~1 turn per pitch along
       the ray, so a side stops 2 turns past k (W - k > 2 outward, < -2 inward): rays without a crossing (wraps leaving the box) had
       scanned the whole ray in dr steps, most of ufsm wraps' time */
#define EVR(R_) evr(ctx, z, cy + (R_) * sy, cx + (R_) * sx, (R_), thw)
    const double h0 = fmin(fmax(r_hint, rlo), rhi), f0 = EVR(h0) - k;
    double ro = h0, fo = f0, ri = h0, fi = f0, best = -1, bestd = 1e300;
    while ((ro < rhi && fo < 2) || (ri > rlo && fi > -2)) {
        if ((!(ro < rhi && fo < 2) || ro > ub) && (!(ri > rlo && fi > -2) || ri < ua)) return -1;   /* only dropped roots are left */
        double seg[RR_SEG][4]; int ns = 0;   /* (a, b, f(a), f(b)) of this ring's crossing intervals */
        if (ro < rhi && fo < 2) {
            const double rn = fmin(ro + dr, rhi), fn = EVR(rn) - k;
            if ((fo < 0) != (fn < 0)) ns = rr_cross(evr, ctx, z, cy, cx, sy, sx, thw, k, ro, fo, rn, fn, 0, seg, ns);
            ro = rn; fo = fn;
        }
        if (ri > rlo && fi > -2) {
            const double rn = fmax(ri - dr, rlo), fn = EVR(rn) - k;
            if ((fi < 0) != (fn < 0)) ns = rr_cross(evr, ctx, z, cy, cx, sy, sx, thw, k, rn, fn, ri, fi, 0, seg, ns);
            ri = rn; fi = fn;
        }
        if (ns) { int out = 1; for (int q = 0; q < ns; q++) out &= seg[q][1] < ua || seg[q][0] > ub; if (out) return -1; }
        for (int q = 0; q < ns; q++) {
            /* Illinois regula falsi to 1e-6 voxel (W is smooth along the ray: ~6 evaluations; the 24 halvings it replaces: dr 2^-24) */
            double a = seg[q][0], b = seg[q][1], fa = seg[q][2], fb = seg[q][3], root = 0.5 * (a + b); int side = 0;   /* (fb: EVR(b) - k, kept) */
            for (int it = 0; it < 60 && b - a > 1e-6; it++) {
                double c = fb != fa ? b - fb * (b - a) / (fb - fa) : 0.5 * (a + b);
                if (!(c > a && c < b)) c = 0.5 * (a + b);
                const double fc = EVR(c) - k; root = c;
                if (fc == 0) { a = b = c; break; }
                if ((fc < 0) == (fb < 0)) { b = c; fb = fc; if (side == -1) fa *= 0.5; side = -1; }
                else { a = c; fa = fc; if (side == 1) fb *= 0.5; side = 1; }
            }
            if (fabs(root - r_hint) < bestd) { bestd = fabs(root - r_hint); best = root; }
        }
        if (ns) break;
    }
    if (best < 0) return -1;
    *r = best;
    return 0;
#undef EVR
}
static double ev_model(const void *m, double z, double y, double x, double r, double th) { return fit_W_rt(m, z, y, x, r, th); }
int fit_ray_root(const fit_model *m, double z, double theta, double k, double r_hint, double *r) { return ray_root(ev_model, nullptr, m, m, z, theta, k, r_hint, r); }

/* plane names of fit_fields_buf a[]: the 10 base planes, the winding mod 4 planes, the recto plane */
static const char *fit_plane_names[17] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "dz", "dy", "dx", "dq", "r", "ct", "cs"};
int fit_fields_save(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[10], const float *cy, const float *cx, int na) {
    uint8_t *b[14] = {0}; for (int k = 0; k < 10; k++) b[k] = a[k];
    return fit_fields_save_ex(dir, origin, shape, h, um, b, 0, nullptr, cy, cx, na);
}
int fit_fields_save_w4(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[14], int w4_d, const float *cy, const float *cx, int na) {
    return fit_fields_save_ex(dir, origin, shape, h, um, a, w4_d, nullptr, cy, cx, na);
}
int fit_fields_save_ex(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[14], int w4_d, const uint8_t *recto,
                       const float *cy, const float *cx, int na) {
    char path[1400]; snprintf(path, sizeof path, "mkdir -p '%s'", dir); if (system(path)) return -1;
    const size_t nf = (size_t)shape[0] * shape[1] * shape[2];
    for (int k = 0; k < 15; k++) {
        const uint8_t *src = k < 14 ? a[k] : recto;
        if (k >= 10 && k < 14 && w4_d <= 0) continue;
        if (!src) { if (k < 10 || k == 14) { if (k < 10) return -1; continue; } return -1; }
        snprintf(path, sizeof path, "%s/%s.raw", dir, fit_plane_names[k]);
        FILE *f = fopen(path, "wb"); if (!f) return -1;
        const int bad = fwrite(src, 1, nf, f) != nf, closed = fclose(f); if (bad || closed) return -1;
    }
    snprintf(path, sizeof path, "%s/fields.json", dir);
    FILE *f = fopen(path, "w"); if (!f) return -1;
    fprintf(f, "{\"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.9g, ", origin[0], origin[1], origin[2], shape[0], shape[1], shape[2], h, um);
    if (w4_d > 0) fprintf(f, "\"w4_d\": %d, ", w4_d);
    fprintf(f, "\"recto\": %s, \"axis\": [", recto ? "true" : "false");
    for (int z = 0; z < na; z++) fprintf(f, "%s[%.3f, %.3f]", z ? ", " : "", cy[z], cx[z]);
    fprintf(f, "]}\n");
    return fclose(f) ? -1 : 0;
}

/* a box of a raw uint8 array of shape rs (z y x) at o, n voxels per axis, row by row; 0 on success */
static int raw_read_box(FILE *f, const long long rs[3], const long long o[3], const int64_t n[3], uint8_t *out) {
    if (o[1] == 0 && o[2] == 0 && n[1] == rs[1] && n[2] == rs[2]) {   /* whole planes */
        const size_t plane = (size_t)rs[1] * rs[2];
        return fseeko(f, (off_t)((size_t)o[0] * plane), SEEK_SET) || fread(out, 1, (size_t)n[0] * plane, f) != (size_t)n[0] * plane;
    }
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) {
        const off_t at = (off_t)((((size_t)(o[0] + z) * rs[1]) + (size_t)(o[1] + y)) * rs[2] + (size_t)o[2]);
        if (fseeko(f, at, SEEK_SET) || fread(out + ((size_t)z * n[1] + y) * n[2], 1, (size_t)n[2], f) != (size_t)n[2]) return -1;
    }
    return 0;
}
/* the array box of a fields.json: "zarr_offset" (zarr) or "raw_offset" + "raw_shape" (raw arrays larger than the box; absent: the
   arrays are the box) */
static void fields_crop(const json *j, int zarr, const int shp[3], long long off[3], long long ashape[3]) {
    for (int d = 0; d < 3; d++) {
        off[d] = (long long)json_num(json_at(json_get(j, zarr ? "zarr_offset" : "raw_offset"), d), 0);
        ashape[d] = zarr ? shp[d] : (long long)json_num(json_at(json_get(j, "raw_shape"), d), shp[d]);
    }
}
/* the volcomp substreams (16-row layers of every 128^3 chunk: bits 4 l .. 4 l + 3 for layer l, z3_read_sub) holding rows [z0, z0 + nz)
   of the chunk rows they touch; 0 (whole chunks) for other chunk shapes or UFSM_FIT_FULLDECODE=1. A slab's 96 rows touch two chunk rows:
   6 layers decoded instead of 16, the same bytes (volcomp's substreams decode independently, byte for byte as a full decode) */
static uint32_t fit_submask(const z3 *za, int64_t z0, int64_t nz) {
    const z3_meta *m = z3_meta_of(za);
    if (m->chunk[0] != 128 || m->chunk[1] != 128 || m->chunk[2] != 128 || nz <= 0 || z0 < 0 || ufsm_env_on("UFSM_FIT_FULLDECODE")) return 0;
    uint32_t mask = 0;
    for (int64_t z = z0; z < z0 + nz;) {
        const int64_t e = (z / 128 + 1) * 128 < z0 + nz ? (z / 128 + 1) * 128 : z0 + nz;
        for (int64_t l = (z % 128) / 16; l <= ((e - 1) % 128) / 16; l++) mask |= 0xFu << (4 * l);
        z = e;
    }
    return mask == 0xffffffffu ? 0 : mask;
}
static json *read_json_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(n + 1); if (fread(t, 1, n, f) != n) { fclose(f); free(t); return nullptr; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); return j;
}
void fit_fields_free(fit_fields_buf *b) { for (int k = 0; k < 17; k++) free(b->a[k]); free(b->cy); free(b->cx); memset(b, 0, sizeof *b); }
/* fields.json "core_line" / "core_side": the half-box mask (src/coreline.h); 0 = none */
static int fields_core(const json *j, coreline *C) {
    const char *cp = json_str(json_get(j, "core_line"), ""); const int side = (int)json_num(json_get(j, "core_side"), 0);
    if (!*cp || !side) return 0;
    if (cl_load(C, cp)) { fprintf(stderr, "fit: cannot load the core line %s\n", cp); return 0; }
    return side < 0 ? -1 : 1;
}
static int g_ct_planes = 3;   /* the CT planes fit_fields_load reads: bit 0 ct, bit 1 cs */
void fit_fields_ct_planes(int mask) { g_ct_planes = mask; }
static int g_skip_planes;   /* optional planes fit_fields_load leaves out: bit 0 the recto plane, bit 1 the winding mod 4 planes */
void fit_fields_skip_planes(int mask) { g_skip_planes = mask; }
int fit_fields_zchunk(const char *dir, long long *zoff) {   /* (src/fit.h) */
    char path[1400]; snprintf(path, sizeof path, "%s/fields.json", dir);
    json *j = read_json_file(path); if (!j) return -1;
    int shp[3]; for (int d = 0; d < 3; d++) shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0);
    const int zarr = !strcmp(json_str(json_get(j, "format"), "raw"), "zarr"); long long zo[3], rsh[3]; fields_crop(j, zarr, shp, zo, rsh); json_free(j);
    int cz = 128; *zoff = 0;
    if (zarr) { z3 *za = z3_open(store_open(dir), "w", nullptr); if (za) { if (z3_meta_of(za)->chunk[0] > 0) cz = z3_meta_of(za)->chunk[0]; z3_close(za); } *zoff = zo[0]; }
    return cz;
}
/* The half-box masks (cl_mask_block) on threads, row by row, with a per-process cache of the cells each row blanks: a fit loads
   every field row three to four times (the prior, the pooled global fit, the slabs with their overlaps), surf-stats the pc / ps / w
   planes of the whole box, and the masks were most of a 2 cm half-box fit's wall time (~11 of 19 min, one core, cl_sdist per cell
   near the core). The first load of a row runs cl_mask_block on it (pc replaced by a zeroed scratch row, whose 128s record the
   blanked cells); later loads of the row set the recorded cells (w = 0, pc = ps = 128). Mode 1 (w given: blocks without material
   skipped, as cl_mask_block does) and mode 0 (no w: every block) are cached apart; a row's blanked cells depend only on the
   row's material, the same on every load of one fields directory. Same arrays as the serial calls, bit for bit. */
typedef struct { char key[1200]; int mode, D, H, W; uint8_t **rows; } mask_cache;
static mask_cache g_mcache[2];
static pthread_mutex_t g_mcache_mu = PTHREAD_MUTEX_INITIALIZER;
static uint8_t **mcache_rows(const char *dir, int mode, int D, int H, int W) {   /* the cache of (dir, mode), reset when the box changes */
    if ((double)D * H * W / 8 > 1.5e9) return nullptr;   /* (bits per cell; huge boxes: no cache, threads only) */
    pthread_mutex_lock(&g_mcache_mu);
    mask_cache *c = &g_mcache[mode];
    if (strcmp(c->key, dir) || c->D != D || c->H != H || c->W != W) {
        if (c->rows) { for (int z = 0; z < c->D; z++) free(c->rows[z]); free(c->rows); }
        snprintf(c->key, sizeof c->key, "%s", dir); c->mode = mode; c->D = D; c->H = H; c->W = W; c->rows = calloc(D, sizeof *c->rows);
    }
    pthread_mutex_unlock(&g_mcache_mu);
    return c->rows;
}
typedef struct { const coreline *C; int side; const long long *org; double h, um; int z0, H, W, Dbox; uint8_t *pc, *ps, *w; uint8_t **cache, **all; } mask_job;
/* the mode-0 cache of dir (every block), when it is there (a w-only load uses it: w & ~bits is cl_mask_block's w, since the blocks it
   skips have no material) */
static uint8_t **mcache_all(const char *dir, int D, int H, int W) {
    pthread_mutex_lock(&g_mcache_mu);
    const mask_cache *c = &g_mcache[0]; uint8_t **r = c->rows && !strcmp(c->key, dir) && c->D == D && c->H == H && c->W == W ? c->rows : nullptr;
    pthread_mutex_unlock(&g_mcache_mu);
    return r;
}
static void mask_rows(void *a, int r0, int r1, int t) {
    mask_job *J = a; const size_t HW = (size_t)J->H * J->W; uint8_t *s = malloc(HW);
    for (int r = r0; r < r1; r++) {
        const int z = J->z0 + r; uint8_t *pc = J->pc ? J->pc + r * HW : nullptr, *ps = J->ps ? J->ps + r * HW : nullptr, *w = J->w ? J->w + r * HW : nullptr;
        uint8_t *bits = J->cache && z >= 0 && z < J->Dbox ? J->cache[z] : nullptr;
        const uint8_t *ab = !bits && J->all && !pc && !ps && w && z >= 0 && z < J->Dbox ? J->all[z] : nullptr;
        if (ab) { for (size_t i = 0; i < HW; i++) if (ab[i >> 3] >> (i & 7) & 1) w[i] = 0; continue; }
        if (!bits) {
            memset(s, 0, HW);
            cl_mask_block(J->C, J->side, J->org, J->h, J->um, z, 1, J->H, J->W, s, ps, w);
            if (J->cache && z >= 0 && z < J->Dbox && (bits = calloc((HW + 7) / 8, 1))) {
                for (size_t i = 0; i < HW; i++) if (s[i] == 128) bits[i >> 3] |= (uint8_t)(1u << (i & 7));
                J->cache[z] = bits;
            }
            if (pc) for (size_t i = 0; i < HW; i++) if (s[i] == 128) pc[i] = 128;
            continue;
        }
        for (size_t i8 = 0; i8 < (HW + 7) / 8; i8++) {
            if (!bits[i8]) continue;
            for (int k = 0; k < 8; k++) {
                const size_t i = i8 * 8 + k; if (i >= HW || !(bits[i8] >> k & 1)) continue;
                if (w) w[i] = 0;
                if (pc) pc[i] = 128;
                if (ps) ps[i] = 128;
            }
        }
    }
    free(s);
}
/* cl_mask_block of rows [z0, z0 + nz) on threads, cached per (dir, mode) (dir null: no cache) */
static void core_mask(const char *dir, const coreline *C, int side, const long long org[3], double h, double um, int Dbox, int z0, int nz, int H, int W,
                      uint8_t *pc, uint8_t *ps, uint8_t *w) {
    mask_job J = {C, side, org, h, um, z0, H, W, Dbox, pc, ps, w, dir ? mcache_rows(dir, w != nullptr, Dbox, H, W) : nullptr,
                  dir && w && !pc && !ps ? mcache_all(dir, Dbox, H, W) : nullptr};
    fit_zpar(nz, fit_nthreads(nz), mask_rows, &J);
}
void fit_core_mask(const coreline *C, int side, const long long org[3], double h, double um, int z0, int nz, int H, int W, uint8_t *pc, uint8_t *ps, uint8_t *w) {
    mask_job J = {C, side, org, h, um, z0, H, W, 0, pc, ps, w, nullptr, nullptr};
    fit_zpar(nz, fit_nthreads(nz), mask_rows, &J);
}
int fit_fields_load(const char *dir, int fz0, int fz1, int pool, fit_fields_buf *b) { return fit_fields_load_ex(dir, fz0, fz1, pool, b, 0, nullptr, 0, nullptr); }
int fit_fields_load_ex(const char *dir, int fz0, int fz1, int pool, fit_fields_buf *b, int noread, fit_fields_buf *also, int also_pool, fit_fields_buf *prev) {   /* (src/fit.h) */
    const char *const *nm = fit_plane_names;
    memset(b, 0, sizeof *b);
    char path[1400]; snprintf(path, sizeof path, "%s/fields.json", dir);
    json *j = read_json_file(path); if (!j) { fprintf(stderr, "fit: cannot read %s\n", path); return -1; }
    int shp[3]; long long org[3];
    for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
    const int h = (int)json_num(json_get(j, "h"), 8), zarr = !strcmp(json_str(json_get(j, "format"), "raw"), "zarr");
    long long zoff[3], rsh[3];   /* "zarr_offset" / "raw_offset" [z, y, x] cells: the box corner inside larger arrays (shape = the box) */
    fields_crop(j, zarr, shp, zoff, rsh);
    b->um = json_num(json_get(j, "um"), 2.4);
    /* the optional planes, unpooled loads only: winding mod 4 (fields.json "w4_d" > 0) and recto ("recto": true requires the r plane,
       false ignores a stale one, absent: used when present) */
    const int w4_d = (int)json_num(json_get(j, "w4_d"), 0);
    const json *jr = json_get(j, "recto"); const int want_recto = !jr ? -1 : (jr->type == J_BOOL || jr->type == J_NUM) ? (jr->num != 0) : -1;
    int act[17], nact = 0, boxraw[17] = {0}; for (int k = 0; k < 10; k++) if (!(g_skip_planes & 4 && pool == 1 && k >= 2 && k < 8) && !(g_skip_planes & 8 && pool == 1 && k == 8)) act[nact++] = k;   /* (bit 2: the normals stay off the host: fit_fields_load_stream; bit 3: no sp, fit-join) */
    if (pool == 1 && w4_d > 0 && !(g_skip_planes & 2)) for (int k = 10; k < 14; k++) act[nact++] = k;
    if (pool == 1 && want_recto != 0 && !(g_skip_planes & 1)) {
        snprintf(path, sizeof path, zarr ? "%s/r/zarr.json" : "%s/r.raw", dir);
        const int exists = access(path, F_OK) == 0;
        if (want_recto > 0 && !exists) { fprintf(stderr, "fit: missing declared recto field %s\n", path); json_free(j); return -1; }
        if (exists) act[nact++] = 14;
    }
    for (int k = 15; k < 17 && pool == 1; k++) {
        if (!(g_ct_planes >> (k - 15) & 1)) continue;   /* (fit_fields_ct_planes: not wanted) */   /* the CT planes: zarr arrays beside the others (predict --fit-zarr --fit-ct), else box-shaped raw
                                                     files (ufsm ct-fields, predict --fit-fields --fit-ct) */
        snprintf(path, sizeof path, "%s/%s/zarr.json", dir, nm[k]);
        if (zarr && access(path, F_OK) == 0) { act[nact++] = k; continue; }
        snprintf(path, sizeof path, "%s/%s.raw", dir, nm[k]);
        if (access(path, F_OK) == 0) { act[nact++] = k; boxraw[k] = 1; }
    }
    if (fz1 <= 0 || fz1 > shp[0]) fz1 = shp[0];
    if (fz0 < 0) fz0 = 0;
    if (pool == 0) {   /* the geometry and the whole axis table only */
        const json *axj = json_get(j, "axis"); const int nat = axj ? (int)axj->n : 0, na = (int)((double)shp[0] * h) + 1;
        b->cy = malloc(na * sizeof(float)); b->cx = malloc(na * sizeof(float));
        for (int z = 0; z < na; z++) { const int zi = z < nat ? z : nat - 1; b->cy[z] = (float)json_num(json_at(json_at(axj, zi), 0), 0); b->cx[z] = (float)json_num(json_at(json_at(axj, zi), 1), 0); }
        json_free(j);
        b->h = h; for (int d = 0; d < 3; d++) { b->shape[d] = shp[d]; b->origin[d] = org[d]; }
        b->f = (fit_fields){shp[0], shp[1], shp[2], (double)h, nullptr, nullptr, {0}, nullptr, nullptr, b->cy, b->cx, {0}, 0, nullptr, nullptr, nullptr};
        return 0;
    }
    if (fz0 >= fz1 || pool < 1) { json_free(j); return -1; }
    const json *axj = json_get(j, "axis"); const int nat = axj ? (int)axj->n : 0;
    coreline core = {0}; const int cside = fields_core(j, &core);
    const int D = fz1 - fz0, H = shp[1], W = shp[2], Dp = (D + pool - 1) / pool, Hp = (H + pool - 1) / pool, Wp = (W + pool - 1) / pool;
    const int na = (int)((double)Dp * pool * h) + 1;   /* the slab's axis table (native z from fz0 h) */
    b->cy = malloc(na * sizeof(float)); b->cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) {
        int zi = fz0 * h + z; if (zi > nat - 1) zi = nat - 1; if (zi < 0) zi = 0;
        b->cy[z] = (float)json_num(json_at(json_at(axj, zi), 0), 0); b->cx[z] = (float)json_num(json_at(json_at(axj, zi), 1), 0);
    }
    json_free(j);
    const size_t np = (size_t)Dp * Hp * Wp, plane = (size_t)H * W;
    int zr0 = fz0;   /* the first row read: past the rows kept from prev */
    if (prev && pool == 1 && !also && !noread && prev->a[9] && prev->h == h && prev->shape[1] == H && prev->shape[2] == W) {
        const long long pz = (prev->origin[0] - org[0]) / h; const int pfz0 = (int)pz, pfz1 = pfz0 + prev->shape[0];
        int same = pz * h == prev->origin[0] - org[0]; for (int k = 0; k < 17; k++) { int in = 0; for (int q = 0; q < nact; q++) in |= act[q] == k; same &= in == (prev->a[k] != nullptr); }
        if (same && pfz0 <= fz0 && fz0 < pfz1 && pfz1 <= fz1) {   /* the overlap rows move to the front of prev's planes (in place) */
            for (int q = 0; q < nact; q++) {
                const int k = act[q]; uint8_t *a = prev->a[k];
                memmove(a, a + (size_t)(fz0 - pfz0) * plane, (size_t)(pfz1 - fz0) * plane);
                uint8_t *r = realloc(a, np); if (!r) { fprintf(stderr, "fit: out of memory\n"); return -1; }
                b->a[k] = r; prev->a[k] = nullptr;
            }
            zr0 = pfz1;
        }
    }
    if (prev) fit_fields_free(prev);
    if (zr0 == fz0) for (int q = 0; q < nact; q++) b->a[act[q]] = calloc(np, 1);
    if (noread) fz1 = fz0;   /* (the loop below reads nothing) */
    /* read in z-blocks of `pool` rows, pooled into one output row each; unpooled (direct), decoded straight into the output rows (no
       block buffers: they were 64 rows x every plane, 8 GB on a full cross-section), in blocks that end on the zarr's chunk rows (each
       chunk decoded once), or of 64 rows when a pooled box is filled alongside */
    const int direct = pool == 1, zb = pool > 1 ? pool : 64;
    uint8_t *blk[17] = {0}; if (!direct) for (int q = 0; q < nact; q++) blk[act[q]] = malloc((size_t)zb * plane);
    z3 *za[17] = {0}; FILE *fr[17] = {0};
    for (int q = 0; q < nact; q++) {
        const int k = act[q];
        if (zarr && !boxraw[k]) { snprintf(path, sizeof path, "%s/%s", dir, nm[k]); za[k] = z3_open(store_open(dir), nm[k], nullptr); }
        else { snprintf(path, sizeof path, "%s/%s.raw", dir, nm[k]); fr[k] = fopen(path, "rb"); }
        if (!za[k] && !fr[k]) { fprintf(stderr, "fit: cannot open %s\n", path); return -1; }
    }
    fit_fields src = {0}; src.h = h;
    int cz = 128; for (int q = 0; q < nact; q++) if (za[act[q]]) { const int c0 = z3_meta_of(za[act[q]])->chunk[0]; if (c0 > 0) cz = c0; break; }
    for (int z0 = zr0, nz = 0; z0 < fz1; z0 += nz) {
        nz = z0 + zb <= fz1 ? zb : fz1 - z0;
        if (direct && !also) { const long long e = zarr ? ((z0 + zoff[0]) / cz + 1) * cz - zoff[0] : (long long)z0 + cz; nz = (int)((e < fz1 ? e : fz1) - z0); }
        if (direct) for (int q = 0; q < nact; q++) blk[act[q]] = b->a[act[q]] + (size_t)(z0 - fz0) * plane;
        for (int q = 0; q < nact; q++) {
            const int k = act[q];
            if (za[k]) { const int64_t o[3] = {z0 + zoff[0], zoff[1], zoff[2]}, n[3] = {nz, H, W}; if (z3_read_sub(za[k], o, n, blk[k], fit_nthreads(16), fit_submask(za[k], o[0], n[0]))) { fprintf(stderr, "fit: %s\n", z3_error()); return -1; } }
            else if (boxraw[k]) { const long long bs[3] = {shp[0], shp[1], shp[2]}, o[3] = {z0, 0, 0}; const int64_t n[3] = {nz, H, W};
                   if (raw_read_box(fr[k], bs, o, n, blk[k])) { fprintf(stderr, "fit: short read of %s\n", nm[k]); return -1; } }
            else { const long long o[3] = {z0 + zoff[0], zoff[1], zoff[2]}; const int64_t n[3] = {nz, H, W};
                   if (raw_read_box(fr[k], rsh, o, n, blk[k])) { fprintf(stderr, "fit: short read\n"); return -1; } }
        }
        if (cside) core_mask(dir, &core, cside, org, h, b->um, shp[0], z0, nz, H, W, blk[0], blk[1], blk[9]);   /* the other side of the core: no data */
        if (pool == 1) {
            for (int g = 0; also && g < nz; g += also_pool) {   /* the pooled box's rows of this block */
                const int gz = g + also_pool <= nz ? also_pool : nz - g;
                src.D = gz; src.H = H; src.W = W; src.pc = blk[0] + g * plane; src.ps = blk[1] + g * plane; for (int t = 0; t < 6; t++) src.t[t] = blk[2 + t] + g * plane;
                src.sp = blk[8] + g * plane; src.w = blk[9] + g * plane;
                fit_pooled pp = fit_pool(&src, also_pool, 0);
                const int zo = (z0 + g) / also_pool; const size_t ap = (size_t)also->shape[1] * also->shape[2];
                for (int k = 0; k < 10; k++) { memcpy(also->a[k] + (size_t)zo * ap, pp.a[k], (size_t)pp.H * pp.W); free(pp.a[k]); }
            }
            continue;
        }
        src.D = nz; src.H = H; src.W = W; src.pc = blk[0]; src.ps = blk[1]; for (int t = 0; t < 6; t++) src.t[t] = blk[2 + t]; src.sp = blk[8]; src.w = blk[9];
        fit_pooled pp = fit_pool(&src, pool, 0);   /* one output row */
        const int zo = (z0 - fz0) / pool;
        for (int k = 0; k < 10; k++) { memcpy(b->a[k] + (size_t)zo * Hp * Wp, pp.a[k], (size_t)pp.H * pp.W); free(pp.a[k]); }
        free(pp.cs); free(pp.pg);
    }
    for (int k = 0; k < 17; k++) { if (!direct) free(blk[k]); if (za[k]) z3_close(za[k]); if (fr[k]) fclose(fr[k]); }
    if (cside) cl_free(&core);
    b->h = h * pool; b->shape[0] = Dp; b->shape[1] = Hp; b->shape[2] = Wp;
    b->origin[0] = org[0] + (long long)fz0 * h; b->origin[1] = org[1]; b->origin[2] = org[2];
    b->f = (fit_fields){Dp, Hp, Wp, (double)b->h, b->a[0], b->a[1], {b->a[2], b->a[3], b->a[4], b->a[5], b->a[6], b->a[7]}, b->a[8], b->a[9], b->cy, b->cx, {0}, 0, nullptr, nullptr, nullptr};
    if (b->a[10]) { for (int k = 0; k < 4; k++) b->f.wd[k] = b->a[10 + k]; b->f.w4_d = w4_d; }
    b->f.recto = b->a[14]; b->f.ct = b->a[15]; b->f.cs = b->a[16];
    return 0;
}

/* ---- streamed fields for GPU fits (src/fit.h fit_fields_load_stream) ---- */
int fit_stream_fps(const fit_cfg *c, int h, int fps[8]) {
    int n = 0;
    for (int st = 0; st < c->nstage && st < c->phase_stage; st++) {
        const int L = c->lattice[st]; int fp = 1; while (2 * fp * (double)h <= L / 4.0) fp *= 2;   /* (fit_run_dir's rule) */
        if (fp <= 1) continue;
        int have = 0; for (int q = 0; q < n; q++) have |= fps[q] == fp;
        if (!have && n < 8) fps[n++] = fp;
    }
    return n;
}
void fit_stream_free(fit_stream *s) {
    if (!s) return;
    if (s->dfield) nn_free(s->dfield);
    for (int q = 0; q < 8; q++) if (s->dpool[q]) nn_free(s->dpool[q]);
    if (s->dcnt) nn_free(s->dcnt);
    memset(s, 0, sizeof *s);
}
int fit_stream_verify(const fit_stream *s, const fit_fields *f) {   /* (src/fit.h) */
    const size_t n = (size_t)f->D * f->H * f->W; if (n != s->n) return 1;
    int bad = 0; uint8_t *h = malloc(n); if (!h) return 1;
    const uint8_t *pl[10] = {f->pc, f->ps, f->t[0], f->t[1], f->t[2], f->t[3], f->t[4], f->t[5], f->sp, f->w};
    for (int k = 0; k < 10; k++) { nn_d2h(h, s->dfield + (size_t)k * n, n); if (memcmp(h, pl[k], n)) { fprintf(stderr, "fit stream verify: device plane %d differs\n", k); bad++; } }
    free(h);
    for (int q = 0; q < s->nfp; q++) {
        fit_pooled pp = fit_pool(f, s->fp[q], 0); const size_t np = (size_t)pp.D * pp.H * pp.W;
        if (pp.D != s->pD[q] || pp.H != s->pH[q] || pp.W != s->pW[q]) bad++;
        else if (!s->dpool[q]) bad++;
        else {
            uint8_t *hp = malloc(np); size_t nz = 0, dc = 0;
            for (int k = 0; k < 10; k++) { nn_d2h(hp, s->dpool[q] + (size_t)k * np, np); if (memcmp(pp.a[k], hp, np)) { fprintf(stderr, "fit stream verify: pooled by %d plane %d differs\n", s->fp[q], k); bad++; } }
            for (size_t i = 0; i < np; i++) nz += pp.a[9][i] != 0;
            unsigned long long cnt[8]; nn_d2h(cnt, s->dcnt, sizeof cnt); dc = cnt[q];
            if (nz != dc) { fprintf(stderr, "fit stream verify: pooled by %d: %zu cells with material, the device counted %zu\n", s->fp[q], nz, dc); bad++; }
            free(hp);
        }
        for (int k = 0; k < 10; k++) free(pp.a[k]);
        free(pp.cs); free(pp.pg);
    }
    return bad;
}
int fit_fields_load_stream(const char *dir, fit_fields_buf *b, fit_stream *s, const int *fps, int nfp, void (*gpu_ready)(void)) {
    memset(s, 0, sizeof *s);
    if (nfp > 8) return -1;
    const int sk = g_skip_planes; g_skip_planes |= 4;   /* the host buffer without the normal planes */
    const int tms = getenv("UFSM_FIT_TIMES") != nullptr; double t0 = nn_wall_seconds(), td = 0, tu = 0;
    const int lr = fit_fields_load(dir, 0, 0, 1, b);
    if (tms) fprintf(stderr, "fit times: load host planes %.2fs\n", nn_wall_seconds() - t0);
    g_skip_planes = sk;
    if (lr) return -1;
    char path[1400]; snprintf(path, sizeof path, "%s/fields.json", dir);
    json *j = read_json_file(path); if (!j) { fit_fields_free(b); return -1; }
    if (gpu_ready) gpu_ready();
    int shp[3]; for (int d = 0; d < 3; d++) shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0);
    const int zarr = !strcmp(json_str(json_get(j, "format"), "raw"), "zarr"); long long zoff[3], rsh[3]; fields_crop(j, zarr, shp, zoff, rsh);
    json_free(j);
    const int D = b->shape[0], H = b->shape[1], W = b->shape[2]; const size_t plane = (size_t)H * W, n = (size_t)D * plane;
    s->n = n;
    {   /* UFSM_FIT_GPU_LOCK=FILE: an exclusive lock on FILE from here (the host planes decoded) to the process's exit, so two fits per GPU
           (one GPU's worth of device memory) overlap one's host decoding with the other's device stages */
        static int locked;   /* (once per process: a second flock of the file from this process would wait on the first) */
        const char *lk = getenv("UFSM_FIT_GPU_LOCK");
        if (lk && *lk && !locked) {
            const double tl = nn_wall_seconds(); const int fd = open(lk, O_RDWR | O_CREAT, 0644); locked = 1;
            if (fd < 0 || flock(fd, LOCK_EX)) fprintf(stderr, "fit: cannot lock %s (going on unlocked)\n", lk);
            else if (tms) fprintf(stderr, "fit times: GPU lock wait %.2fs\n", nn_wall_seconds() - tl);   /* (fd kept open: released at exit) */
        }
    }
    const int force_managed = ufsm_env_on("UFSM_FIT_MANAGED");
    /* on the device only with room beside them for the pooled stages, g_fit_dev_reserve (the ref models, the phase gate's blocks) and
       512 MB of fit buffers: fields that fit alone left the pooled planes' allocations to fail (null -> illegal access, desk 0490B 96x3916x4096) */
    size_t pooled = 0; for (int q = 0; q < nfp; q++) { const size_t f = (size_t)fps[q]; pooled += 10 * ((D + f - 1) / f) * ((H + f - 1) / f) * ((W + f - 1) / f); }
    const size_t need = 10 * n + pooled + g_fit_dev_reserve + ((size_t)512 << 20), fr = force_managed ? 0 : nn_mem_free();
    s->dfield = force_managed || fr < need ? nullptr : nn_malloc_try(10 * n);
    if (!s->dfield) {
        s->dfield = nn_managed_alloc(10 * n);
        if (s->dfield) fprintf(stderr, "fit: fields in managed memory (%.1f GB, %s)\n", 10.0 * n / 1e9, force_managed ? "UFSM_FIT_MANAGED" : "over the free device memory");
        if (s->dfield && !force_managed) fprintf(stderr, "fit: (device %.2f GB free < fields %.2f + pooled %.2f + reserve %.2f + 0.5 GB)\n", fr / 1e9, 10.0 * n / 1e9, pooled / 1e9, g_fit_dev_reserve / 1e9);
    }
    if (!s->dfield) { fprintf(stderr, "fit: no device memory for the fields (%.1f GB: 10 B per cell of %dx%dx%d)\n", 10.0 * n / 1e9, D, H, W); fit_fields_free(b); return -1; }
    /* the host planes, then the normals (per plane, blocks ending on the zarr's chunk rows, two block buffers in turn) to the device on
       the upload thread, beside the decoding */
    nn_fit_up *up = nn_fit_up_open();
    const int hk[4] = {0, 1, 8, 9}; for (int q = 0; q < 4; q++) nn_fit_up_put(up, s->dfield + (size_t)hk[q] * n, b->a[hk[q]], n);
    int cz = 128, nb = 0; uint8_t *buf[2] = {nullptr, nullptr}; size_t bcap[2] = {0, 0}; long bid[2] = {-1, -1}; int rc = 0;
    for (int k = 2; k < 8 && !rc; k++) {
        z3 *za = nullptr; FILE *fr = nullptr;
        if (zarr) { za = z3_open(store_open(dir), fit_plane_names[k], nullptr); if (za && z3_meta_of(za)->chunk[0] > 0) cz = z3_meta_of(za)->chunk[0]; }
        else { snprintf(path, sizeof path, "%s/%s.raw", dir, fit_plane_names[k]); fr = fopen(path, "rb"); }
        if (!za && !fr) { fprintf(stderr, "fit: cannot open the %s plane of %s\n", fit_plane_names[k], dir); rc = -1; break; }
        for (int z0 = 0, nz = 0; z0 < D && !rc; z0 += nz, nb ^= 1) {
            const long long e = zarr ? ((z0 + zoff[0]) / cz + 1) * cz - zoff[0] : (long long)z0 + cz; nz = (int)((e < D ? e : D) - z0);
            double tb = nn_wall_seconds();
            if (bid[nb] >= 0) nn_fit_up_wait(up, bid[nb]);   /* (the buffer's previous block uploaded) */
            tu += nn_wall_seconds() - tb;
            if ((size_t)nz * plane > bcap[nb]) { free(buf[nb]); bcap[nb] = (size_t)nz * plane; buf[nb] = malloc(bcap[nb]); if (!buf[nb]) { rc = -1; break; } }
            const long long o[3] = {z0 + zoff[0], zoff[1], zoff[2]}; const int64_t nn3[3] = {nz, H, W};
            tb = nn_wall_seconds();
            if (za) { const int64_t oi[3] = {o[0], o[1], o[2]}; if (z3_read_sub(za, oi, nn3, buf[nb], fit_nthreads(16), fit_submask(za, oi[0], nn3[0]))) { fprintf(stderr, "fit: %s\n", z3_error()); rc = -1; } }
            else if (raw_read_box(fr, rsh, o, nn3, buf[nb])) { fprintf(stderr, "fit: short read of %s\n", fit_plane_names[k]); rc = -1; }
            td += nn_wall_seconds() - tb;
            if (!rc) bid[nb] = nn_fit_up_put(up, s->dfield + (size_t)k * n + (size_t)z0 * plane, buf[nb], (size_t)nz * plane);
        }
        if (za) z3_close(za);
        if (fr) fclose(fr);
    }
    const double tc = nn_wall_seconds(); nn_fit_up_close(up); tu += nn_wall_seconds() - tc;
    free(buf[0]); free(buf[1]);
    if (tms) fprintf(stderr, "fit times: normals decode %.2fs, upload waits %.2fs\n", td, tu);
    t0 = nn_wall_seconds();
    /* the gradient stages' pooled fields, pooled on the device (nn_fit_pool: fit_pool's sums and roundings; the normals stay there) */
    if (nfp && !rc) { if (!(s->dcnt = nn_malloc_try(8 * sizeof *s->dcnt))) rc = -1; else nn_zero(s->dcnt, 8 * sizeof *s->dcnt); }
    for (int q = 0; q < nfp && !rc; q++) {
        const int fp = fps[q]; s->fp[q] = fp; s->pD[q] = (D + fp - 1) / fp; s->pH[q] = (H + fp - 1) / fp; s->pW[q] = (W + fp - 1) / fp;
        const size_t pb = 10 * (size_t)s->pD[q] * s->pH[q] * s->pW[q];
        if (!(s->dpool[q] = nn_malloc_try(pb))) { fprintf(stderr, "fit: no device memory for the fields pooled by %d (%.2f GB; UFSM_FIT_MANAGED=1: the fields in managed memory)\n", fp, pb / 1e9); rc = -1; break; }
        nn_fit_pool(s->dfield, D, H, W, (double)b->h, fp, s->dpool[q], s->dcnt + q);
    }
    s->nfp = rc ? 0 : nfp;
    if (tms) fprintf(stderr, "fit times: pooled stages %.2fs\n", nn_wall_seconds() - t0);
    if (rc) { fit_stream_free(s); fit_fields_free(b); return -1; }
    return 0;
}

/* ---- a fit set: the global model plus z-slab models (ufsm fit --slab) ---- */
const fit_model *fit_set_model(const fit_set *s, double z, double *zoff) {   /* the model whose interior holds box-local z */
    for (int k = 0; k < s->n; k++) if (z >= s->z0[k] && z < s->z1[k]) { *zoff = (double)(s->slab[k]->origin[0] - s->global->origin[0]); return s->slab[k]; }
    *zoff = 0; return s->global;
}
static double slab_W(const fit_set *s, int k, double z, double y, double x) { return fit_W(s->slab[k], z - (double)(s->slab[k]->origin[0] - s->global->origin[0]), y, x); }
double fit_set_W(const fit_set *s, double z, double y, double x) {   /* blended linearly over +-blend around each slab boundary */
    for (int k = 0; k < s->n; k++) {
        if (z < s->z0[k] || z >= s->z1[k]) continue;
        if (k > 0 && s->blend > 0 && z < s->z0[k] + s->blend) { const double t = (z - s->z0[k] + s->blend) / (2 * s->blend); return (1 - t) * slab_W(s, k - 1, z, y, x) + t * slab_W(s, k, z, y, x); }
        if (k + 1 < s->n && s->blend > 0 && z >= s->z1[k] - s->blend) { const double t = (z - s->z1[k] + s->blend) / (2 * s->blend); return (1 - t) * slab_W(s, k, z, y, x) + t * slab_W(s, k + 1, z, y, x); }
        return slab_W(s, k, z, y, x);
    }
    return fit_W(s->global, z, y, x);
}
static double slab_W_rt(const fit_set *s, int k, double z, double y, double x, double r, double th) {
    return fit_W_rt(s->slab[k], z - (double)(s->slab[k]->origin[0] - s->global->origin[0]), y, x, r, th);
}
static double ev_set(const void *sv, double z, double y, double x, double r, double th) {   /* fit_set_W on a ray (the slabs share the axis) */
    const fit_set *s = sv;
    for (int k = 0; k < s->n; k++) {
        if (z < s->z0[k] || z >= s->z1[k]) continue;
        if (k > 0 && s->blend > 0 && z < s->z0[k] + s->blend) { const double t = (z - s->z0[k] + s->blend) / (2 * s->blend); return (1 - t) * slab_W_rt(s, k - 1, z, y, x, r, th) + t * slab_W_rt(s, k, z, y, x, r, th); }
        if (k + 1 < s->n && s->blend > 0 && z >= s->z1[k] - s->blend) { const double t = (z - s->z1[k] + s->blend) / (2 * s->blend); return (1 - t) * slab_W_rt(s, k, z, y, x, r, th) + t * slab_W_rt(s, k + 1, z, y, x, r, th); }
        return slab_W_rt(s, k, z, y, x, r, th);
    }
    return fit_W_rt(s->global, z, y, x, r, th);
}
double fit_set_gradW(const fit_set *s, double z, double y, double x, double e, double g[3]) {
    const fit_model *m = s->global; const int na = (int)(m->D * m->h) + 1;
    const double th0 = atan2(y - axis_lerp(m->cy, na, z), x - axis_lerp(m->cx, na, z));
    double l2 = 0;
    for (int d = 0; d < 3; d++) {
        double Wv[2];
        for (int k = 0; k < 2; k++) {
            const double sg = k ? e : -e, pz = z + (d == 0 ? sg : 0), py = y + (d == 1 ? sg : 0), px = x + (d == 2 ? sg : 0);
            const double dy = py - axis_lerp(m->cy, na, pz), dx = px - axis_lerp(m->cx, na, pz);
            double th = atan2(dy, dx) - th0; th -= 2 * M_PI * floor((th + M_PI) / (2 * M_PI));
            Wv[k] = ev_set(s, pz, py, px, hypot(dy, dx), th0 + th);
        }
        g[d] = (Wv[1] - Wv[0]) / (2 * e); l2 += g[d] * g[d];
    }
    return sqrt(l2);
}
typedef struct { int mode; double t; ev_part a, b; double tta, ttb; } ev_row;   /* mode 0: a alone; 1: (1 - t) a + t b; tt*: the ray's angle terms */
static double ev_row_W(const void *rv, double z, double y, double x, double r, double th) {
    const ev_row *R = rv;
    return R->mode ? (1 - R->t) * ev_part_W(&R->a, y, x, r, th) + R->t * ev_part_W(&R->b, y, x, r, th) : ev_part_W(&R->a, y, x, r, th);
}
/* ev_row_W along one ray: its angle terms dir th / 2 pi computed once (ev_row_prep; the same expression, so the same W bits) */
static void ev_row_prep(const void *rv, double th) {
    ev_row *R = (ev_row *)rv;   /* (the caller's own row context) */
    R->tta = R->a.m->dir * th / (2 * M_PI); if (R->mode) R->ttb = R->b.m->dir * th / (2 * M_PI);
}
static double ev_row_Wt(const void *rv, double z, double y, double x, double r, double th) {
    const ev_row *R = rv;
    return R->mode ? (1 - R->t) * (ev_part_U(&R->a, y, x, r) - R->tta) + R->t * (ev_part_U(&R->b, y, x, r) - R->ttb) : ev_part_U(&R->a, y, x, r) - R->tta;
}
static void ev_row_init(ev_row *R, const fit_set *s, double z) {   /* ev_set's choice at z */
    R->mode = 0; R->t = 0;
#define ZL(k_) (z - (double)(s->slab[k_]->origin[0] - s->global->origin[0]))
    for (int k = 0; k < s->n; k++) {
        if (z < s->z0[k] || z >= s->z1[k]) continue;
        if (k > 0 && s->blend > 0 && z < s->z0[k] + s->blend) { R->mode = 1; R->t = (z - s->z0[k] + s->blend) / (2 * s->blend); ev_part_init(&R->a, s->slab[k - 1], ZL(k - 1)); ev_part_init(&R->b, s->slab[k], ZL(k)); return; }
        if (k + 1 < s->n && s->blend > 0 && z >= s->z1[k] - s->blend) { R->mode = 1; R->t = (z - s->z1[k] + s->blend) / (2 * s->blend); ev_part_init(&R->a, s->slab[k], ZL(k)); ev_part_init(&R->b, s->slab[k + 1], ZL(k + 1)); return; }
        ev_part_init(&R->a, s->slab[k], ZL(k)); return;
    }
#undef ZL
    ev_part_init(&R->a, s->global, z);
}
int fit_set_ray_root(const fit_set *s, double z, double theta, double k, double r_hint, double *r) {
    ev_row R; ev_row_init(&R, s, z);
    return ray_root(ev_row_Wt, ev_row_prep, &R, s->global, z, theta, k, r_hint, r);
}
int fit_set_ray_root_b(const fit_set *s, double z, double theta, double k, double r_hint, double ua, double ub, double *r) {
    ev_row R; ev_row_init(&R, s, z);
    return ray_root_b(ev_row_Wt, ev_row_prep, &R, s->global, z, theta, k, r_hint, ua, ub, r);
}
/* the same for many rays of one z (ufsm wraps' rows): the slab choice and the z parts once per row */
struct fit_set_row { const fit_set *s; double z; ev_row R; };
fit_set_row *fit_set_row_new(const fit_set *s, double z) { fit_set_row *w = malloc(sizeof *w); if (w) { w->s = s; w->z = z; ev_row_init(&w->R, s, z); } return w; }
int fit_set_row_root_b(fit_set_row *w, double theta, double k, double r_hint, double ua, double ub, double *r) {
    return ray_root_b(ev_row_Wt, ev_row_prep, &w->R, w->s->global, w->z, theta, k, r_hint, ua, ub, r);
}
void fit_set_row_free(fit_set_row *w) { free(w); }
int fit_set_save(const fit_set *s, const char *dir) {
    char p[1400]; snprintf(p, sizeof p, "%s/global", dir);
    if (fit_save(s->global, p)) return -1;
    snprintf(p, sizeof p, "%s/set.json", dir);
    FILE *f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "{\"global\": \"global\", \"slabs\": [");
    for (int k = 0; k < s->n; k++) {
        char sd[1400]; snprintf(sd, sizeof sd, "%s/slab%03d", dir, k);
        if (fit_save(s->slab[k], sd)) { fclose(f); return -1; }
        fprintf(f, "%s{\"dir\": \"slab%03d\", \"z0\": %.3f, \"z1\": %.3f}", k ? ", " : "", k, s->z0[k], s->z1[k]);
    }
    fprintf(f, "], \"blend\": %.3f}\n", s->blend);
    return fclose(f) ? -1 : 0;
}
typedef struct { fit_set *s; const char (*paths)[1400]; _Atomic int next; } fsl_job;
static void *fsl_worker(void *arg) {
    fsl_job *L = arg;
    for (int k; (k = atomic_fetch_add(&L->next, 1)) < L->s->n;) L->s->slab[k] = fit_load(L->paths[k]);
    return nullptr;
}
fit_set *fit_set_load(const char *dir) {
    char p[1400]; snprintf(p, sizeof p, "%s/set.json", dir);
    fit_set *s = calloc(1, sizeof *s);
    json *j = read_json_file(p);
    if (!j) {   /* a single model directory */
        s->global = fit_load(dir);
        if (!s->global) { free(s); return nullptr; }
        return s;
    }
    snprintf(p, sizeof p, "%s/%s", dir, json_str(json_get(j, "global"), "global"));
    s->global = fit_load(p); s->blend = json_num(json_get(j, "blend"), 0);
    const json *sl = json_get(j, "slabs"); s->n = sl ? (int)sl->n : 0;
    s->slab = calloc(s->n + 1, sizeof *s->slab); s->z0 = calloc(s->n + 1, sizeof(double)); s->z1 = calloc(s->n + 1, sizeof(double));
    char (*paths)[1400] = malloc((size_t)(s->n + 1) * sizeof *paths);
    for (int k = 0; k < s->n; k++) {
        const json *e = json_at(sl, k);
        snprintf(paths[k], sizeof paths[k], "%s/%s", dir, json_str(json_get(e, "dir"), ""));
        s->z0[k] = json_num(json_get(e, "z0"), 0); s->z1[k] = json_num(json_get(e, "z1"), 0);
    }
    json_free(j);
    /* the slab models on threads (a whole-scroll join: ~80 slabs, 5 GB of deltas read one after another took ~10 s of ufsm wraps) */
    fsl_job L = {s, (const char (*)[1400])paths, 0};
    pthread_t th[16]; const int nt = s->n < 2 ? 0 : s->n < 16 ? s->n : 16; int up = 0;
    for (; up < nt; up++) if (pthread_create(&th[up], nullptr, fsl_worker, &L)) break;
    fsl_worker(&L);
    for (int t = 0; t < up; t++) pthread_join(th[t], nullptr);
    free(paths);
    for (int k = 0; k < s->n; k++) if (!s->slab[k]) { fit_set_free(s); return nullptr; }
    if (!s->global) { fit_set_free(s); return nullptr; }
    return s;
}
void fit_set_free(fit_set *s) {
    if (!s) return;
    fit_free(s->global); for (int k = 0; k < s->n; k++) fit_free(s->slab[k]);
    free(s->slab); free(s->z0); free(s->z1); free(s);
}

/* Slab driver: the global coarse fit on fields pooled to ~64 native voxels (gradient terms, lattices 512 / 256 / 128), then per
   z-slab of slab native voxels (+ overlap on both sides) the fine fit (lattices 64 / 32, phase from the start with the voxel
   unwrapping) started from the global model; each slab's integer gauge is aligned to the previous slab's on their overlap. */
fit_set *fit_run_slabs(const char *dir, const fit_cfg *c, int slab, int overlap) {
    const int tm = getenv("UFSM_FIT_TIMES") != nullptr; double t0 = nn_wall_seconds(), tl = 0;   /* UFSM_FIT_TIMES: wall time per part */
#define FIT_TM(...) do { if (tm) { fprintf(stderr, "fit times: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, " %.2fs\n", nn_wall_seconds() - t0); t0 = nn_wall_seconds(); } } while (0)
    fit_fields_buf all; char p[1400]; snprintf(p, sizeof p, "%s/fields.json", dir);
    json *j = read_json_file(p); if (!j) return nullptr;
    const int h = (int)json_num(json_get(j, "h"), 8), D = (int)json_num(json_at(json_get(j, "shape"), 0), 0);
    json_free(j);
    const double fu = 2.4 / (c->um > 0 ? c->um : 2.4);   /* lattices and pooling in 2.4 um voxels -> native */
    int pool = 1; while (2 * pool * h <= 64 * fu) pool *= 2;
    /* the pooled box is filled by the prior's full-resolution loads below (each field row read and masked once, not twice) when the
       prior's slabs start on pooled rows */
    const int merge = slab % (h * pool) == 0;
    if (merge ? fit_fields_load_ex(dir, 0, 0, pool, &all, 1, nullptr, 0, nullptr) : fit_fields_load(dir, 0, 0, pool, &all)) return nullptr;
    /* the prior from full-resolution rows, slab by slab: knots at the global knot positions (slabs start on them), offsets made
       continuous across slabs, the direction from the summed ring votes */
    const int nkg = (int)ceil((double)all.shape[0] * all.h / c->knot_dz) + 1;
    fit_model P = {0}; P.knot_dz = c->knot_dz; P.nk = nkg < 2 ? 2 : nkg;
    P.pitch = calloc(P.nk, sizeof(float)); P.offset = calloc(P.nk, sizeof(float)); uint8_t *have = calloc(P.nk, 1);
    double vote = 0;
    if (slab % c->knot_dz) { fprintf(stderr, "fit slabs: --slab must be a multiple of the knot spacing %d\n", c->knot_dz); fit_fields_free(&all); return nullptr; }
    for (int z0 = 0; z0 < D * h; z0 += slab) {
        fit_fields_buf cb; const int fz0 = z0 / h, fz1 = (int)fmin(D, ceil((double)(z0 + slab) / h));
        const double tq = nn_wall_seconds();
        if (fit_fields_load_ex(dir, fz0, fz1, 1, &cb, 0, merge ? &all : nullptr, pool, nullptr)) { fit_fields_free(&all); return nullptr; }
        tl += nn_wall_seconds() - tq;
        fit_model tmp = {0}; fit_cfg cp = *c; cp.verbose = 0;
        if (!fit_init_prior(&cb.f, &cp, &tmp)) {
            vote += tmp.dir * tmp.dir_vote;
            for (int k = 0; k < tmp.nk; k++) { const int kg = z0 / c->knot_dz + k; if (kg < P.nk && !have[kg]) { P.pitch[kg] = tmp.pitch[k]; P.offset[kg] = tmp.offset[k]; have[kg] = 1; } }
        }
        free(tmp.pitch); free(tmp.offset); fit_fields_free(&cb);
    }
    { int first = -1; for (int k = 0; k < P.nk; k++) if (have[k]) { first = k; break; }
      if (first < 0) { fprintf(stderr, "fit slabs: no data for the prior\n"); fit_fields_free(&all); return nullptr; }
      double po = P.offset[first], pp = P.pitch[first];
      for (int k = 0; k < P.nk; k++) { if (!have[k]) { P.pitch[k] = (float)pp; P.offset[k] = (float)po; } P.offset[k] += (float)round(po - P.offset[k]); po = P.offset[k]; pp = P.pitch[k]; } }
    free(have);
    P.dir = c->dir ? c->dir : (vote > 0 ? 1 : -1);   /* the prior's dir_vote sign convention: dir * |vote| */
    P.L = 1 << 20; P.lz = P.ly = P.lx = 2; P.delta = calloc(8, sizeof(float));   /* delta 0 */
    P.D = all.shape[0]; P.H = all.shape[1]; P.W = all.shape[2]; P.h = all.h; P.cy = all.cy; P.cx = all.cx;
    if (c->verbose) fprintf(stderr, "fit slabs: prior dir %+d (vote %.2f), pitch %.1f..%.1f over %d knots\n", P.dir, vote, P.pitch[0], P.pitch[P.nk - 1], P.nk);
    FIT_TM("prior (field loads %.2fs, %s)", tl, merge ? "the pooled box filled by them" : "the pooled box loaded apart");
    fit_cfg cg = *c; cg.nstage = 3; cg.lattice[0] = fit_scale_L(512, fu); cg.lattice[1] = fit_scale_L(256, fu); cg.lattice[2] = fit_scale_L(128, fu); cg.phase_stage = 3; cg.init = &P; cg.init_off[0] = cg.init_off[1] = cg.init_off[2] = 0; cg.dir = P.dir;
    for (int k = 0; k < 3; k++) cg.steps[k] = c->steps[0];
    cg.w_recto = 0;   /* pooled fields carry no recto plane (the slabs fit the anchors) */
    cg.w_ct = 0;      /* nor CT planes (the slabs fit them) */
    if (c->verbose) fprintf(stderr, "fit slabs: global coarse fit on fields pooled by %d (%dx%dx%d, h %d)\n", pool, all.shape[0], all.shape[1], all.shape[2], all.h);
    fit_set *s = calloc(1, sizeof *s); s->blend = overlap / 2.0;
    s->global = fit_run(&all.f, &cg);
    free(P.pitch); free(P.offset); free(P.delta);
    if (!s->global) { fit_fields_free(&all); free(s); return nullptr; }
    for (int d = 0; d < 3; d++) s->global->origin[d] = all.origin[d];
    fit_fields_free(&all);
    FIT_TM("global coarse fit");
    const double Z = (double)D * h; const int ns = (int)ceil(Z / slab);
    s->slab = calloc(ns, sizeof *s->slab); s->z0 = calloc(ns, sizeof(double)); s->z1 = calloc(ns, sizeof(double));
    fit_fields_buf prevb; memset(&prevb, 0, sizeof prevb);   /* the previous slab's fields: its overlap rows are kept, not read again */
    for (int k = 0; k < ns; k++) {
        const double zi0 = (double)k * slab, zi1 = fmin(Z, zi0 + slab);
        const int fz0 = (int)fmax(0, floor((zi0 - overlap) / h)), fz1 = (int)fmin(D, ceil((zi1 + overlap) / h));
        fit_fields_buf sb;
        if (fit_fields_load_ex(dir, fz0, fz1, 1, &sb, 0, nullptr, 0, &prevb)) { fit_set_free(s); return nullptr; }
        FIT_TM("slab %d fields", k + 1);
        fit_cfg cs = *c; cs.init = s->global; cs.init_off[0] = (double)fz0 * h; cs.init_off[1] = cs.init_off[2] = 0; cs.dir = s->global->dir;
        cs.emb_zoff = c->emb_zoff + (double)fz0 * h; cs.fib_zoff = c->fib_zoff + (double)fz0 * h;
        if (k > 0) {   /* the flood continues from the previous slab's interior (the overlap below zi0) */
            cs.seed = s->slab[k - 1]; cs.seed_off = (double)(fz0 * h) - (double)(s->slab[k - 1]->origin[0] - s->global->origin[0]); cs.seed_z = zi0 - (double)fz0 * h;
        }
        cs.nstage = 2; cs.lattice[0] = fit_scale_L(64, fu); cs.lattice[1] = fit_scale_L(32, fu); cs.phase_stage = 0; cs.steps[0] = cs.steps[1] = c->steps[c->nstage > 4 ? 4 : 0];
        if (c->verbose) fprintf(stderr, "fit slabs: slab %d of %d, native z [%.0f, %.0f) with fields [%d, %d)\n", k + 1, ns, zi0, zi1, fz0, fz1);
        fit_model *m = fit_run(&sb.f, &cs);
        if (!m) { fit_fields_free(&sb); fit_set_free(s); return nullptr; }
        for (int d = 0; d < 3; d++) m->origin[d] = sb.origin[d];
        if (k > 0) {   /* the integer gauge: the median rounded difference to the previous slab on the overlap */
            const fit_model *pm = s->slab[k - 1]; int hist[21] = {0};
            const double zlo = fmax(zi0 - overlap, 0), zhi = fmin(zi0 + overlap, Z), Y = sb.shape[1] * (double)h, X = sb.shape[2] * (double)h;
            for (int t = 0; t < 4000; t++) {
                const double z = zlo + (zhi - zlo) * ((t * 0.6180339887) - floor(t * 0.6180339887)), y = Y * ((t * 0.7548776662) - floor(t * 0.7548776662)), x = X * ((t * 0.5698402910) - floor(t * 0.5698402910));
                const long dk = lround(fit_W(m, z - (m->origin[0] - s->global->origin[0]), y, x) - fit_W(pm, z - (pm->origin[0] - s->global->origin[0]), y, x));
                if (dk >= -10 && dk <= 10) hist[dk + 10]++;
            }
            int best = 10; for (int q = 0; q < 21; q++) if (hist[q] > hist[best]) best = q;
            const int shift = best - 10;
            if (shift) { for (int q = 0; q < m->nk; q++) m->offset[q] -= (float)shift; if (c->verbose) fprintf(stderr, "fit slabs: slab %d gauge shifted by %d to match slab %d\n", k + 1, -shift, k); }
        }
        s->slab[k] = m; s->z0[k] = zi0; s->z1[k] = zi1; s->n = k + 1;
        prevb = sb;
        FIT_TM("slab %d fit", k + 1);
    }
    fit_fields_free(&prevb);
#undef FIT_TM
    return s;
}

/* ---- grid evaluation of a fit set on the GPU ---- */
struct fit_set_dev { int nm; fit_dev *hm, *dm; void **buf; int nb; float *out; size_t outn; fit_grid_row *rows; int nrows; };
/* (nn_malloc_try: a device short of memory leaves no sticky error; a null pointer here once became an illegal access in nn_fit_grid) */
static void *dev_copy(fit_set_dev *g, const void *src, size_t n, int *bad) { if (*bad) return nullptr; void *d = nn_malloc_try(n ? n : 4); if (!d) { *bad = 1; return nullptr; } if (n) nn_h2d(d, src, n); g->buf[g->nb++] = d; return d; }
fit_set_dev *fit_set_upload(const fit_set *s) {
    if (s->n + 1 > 4096) { fprintf(stderr, "fit_set_upload: at most 4095 slabs\n"); return nullptr; }
    fit_set_dev *g = calloc(1, sizeof *g); g->nm = s->n + 1; g->hm = calloc(g->nm, sizeof(fit_dev)); g->buf = calloc((size_t)5 * g->nm + 1, sizeof *g->buf);   /* (5 arrays per model, the model table) */
    int bad = 0; size_t need = 0;
    for (int k = 0; k < g->nm; k++) {
        const fit_model *m = k < s->n ? s->slab[k] : s->global; fit_dev *d = &g->hm[k];
        const int na = (int)(m->D * m->h) + 1;
        d->D = m->D; d->H = m->H; d->W = m->W; d->h = (float)m->h;
        d->cy = dev_copy(g, m->cy, na * 4, &bad); d->cx = dev_copy(g, m->cx, na * 4, &bad); d->naxis = na;
        d->dir = m->dir; d->nk = m->nk; d->knot_dz = m->knot_dz; d->pitch = dev_copy(g, m->pitch, m->nk * 4, &bad); d->offset = dev_copy(g, m->offset, m->nk * 4, &bad);
        d->L = m->L; d->lz = m->lz; d->ly = m->ly; d->lx = m->lx; d->delta = dev_copy(g, m->delta, (size_t)m->lz * m->ly * m->lx * 4, &bad);
    }
    g->dm = dev_copy(g, g->hm, g->nm * sizeof(fit_dev), &bad);
    if (bad) {   /* (callers fall back to the CPU) */
        for (int k = 0; k < s->n + 1; k++) { const fit_model *m = k < s->n ? s->slab[k] : s->global; need += (size_t)m->lz * m->ly * m->lx * 4; }
        fprintf(stderr, "fit_set_upload: no device memory for the %.2f GB of models\n", need / 1e9); fit_set_dev_free(g); return nullptr;
    }
    return g;
}
void fit_set_dev_free(fit_set_dev *g) {
    if (!g) return;
    for (int k = 0; k < g->nb; k++) nn_free(g->buf[k]);
    if (g->out) nn_free(g->out);
    if (g->rows) nn_free(g->rows);
    free(g->hm); free(g->buf); free(g);
}
static fit_grid_row set_row(const fit_set *s, double z) {   /* the models of box-local z, as fit_set_W */
    fit_grid_row r = {s->n, s->n, (float)z, 0.f, 0.f, 0.f};
    for (int k = 0; k < s->n; k++) {
        if (z < s->z0[k] || z >= s->z1[k]) continue;
        r.a = r.b = k;
        if (k > 0 && s->blend > 0 && z < s->z0[k] + s->blend) { r.a = k - 1; r.b = k; r.t = (float)((z - s->z0[k] + s->blend) / (2 * s->blend)); }
        else if (k + 1 < s->n && s->blend > 0 && z >= s->z1[k] - s->blend) { r.a = k; r.b = k + 1; r.t = (float)((z - s->z1[k] + s->blend) / (2 * s->blend)); }
        break;
    }
    r.za = r.a < s->n ? (float)(s->slab[r.a]->origin[0] - s->global->origin[0]) : 0.f;
    r.zb = r.b < s->n ? (float)(s->slab[r.b]->origin[0] - s->global->origin[0]) : 0.f;
    return r;
}
int fit_set_grid(fit_set_dev *g, const fit_set *s, const double o[3], const int64_t n[3], double step, int with_theta, float *out) {
    const size_t per = (size_t)n[1] * n[2]; if (!per || n[0] <= 0) return 0;
    int64_t zb = (int64_t)((64u << 20) / per); if (zb < 1) zb = 1; if (zb > n[0]) zb = n[0];   /* ~256 MB of output per batch */
    if (g->outn < (size_t)zb * per) { if (g->out) nn_free(g->out); g->outn = (size_t)zb * per; g->out = nn_malloc(g->outn * 4); if (!g->out) return -1; }
    if (g->nrows < zb) { if (g->rows) nn_free(g->rows); g->nrows = 0; if (!(g->rows = nn_malloc_try(zb * sizeof(fit_grid_row)))) return -1; g->nrows = (int)zb; }
    fit_grid_row *hr = malloc(zb * sizeof *hr);
    for (int64_t z0 = 0; z0 < n[0]; z0 += zb) {
        const int64_t nz = n[0] - z0 < zb ? n[0] - z0 : zb;
        for (int64_t a = 0; a < nz; a++) hr[a] = set_row(s, o[0] + ((double)(z0 + a) + 0.5) * step);
        nn_h2d(g->rows, hr, nz * sizeof *hr);
        nn_fit_grid(g->dm, g->rows, (int)nz, (int)n[1], (int)n[2], (float)(o[1]), (float)(o[2]), (float)step, with_theta, g->out);
        nn_d2h(out + (size_t)z0 * per, g->out, (size_t)nz * per * 4);
    }
    free(hr);
    return 0;
}

/* fit_set_grid on the CPU: threads over the z rows; mask (optional, laid out as out): only points whose mask value is 1..252 (a band
   code) are evaluated, the others are left 0 */
typedef struct { const fit_set *s; const double *o; const int64_t *n; double step; int with_theta; const uint8_t *mask; float *out; } grid_cpu_job;
static void grid_cpu_rows(void *ctx, int z0, int z1, int t_) {
    const grid_cpu_job *J = ctx; const fit_model *g = J->s->global; const int na = (int)(g->D * g->h) + 1;
    for (int a = z0; a < z1; a++) {
        const double z = J->o[0] + (a + 0.5) * J->step, cy = axis_lerp(g->cy, na, z), cx = axis_lerp(g->cx, na, z);
        for (int64_t b = 0; b < J->n[1]; b++) for (int64_t c = 0; c < J->n[2]; c++) {
            const size_t i = ((size_t)a * J->n[1] + b) * J->n[2] + c;
            if (J->mask && (J->mask[i] < 1 || J->mask[i] > 252)) { J->out[i] = 0; continue; }
            const double y = J->o[1] + (b + 0.5) * J->step, x = J->o[2] + (c + 0.5) * J->step;
            double W = fit_set_W(J->s, z, y, x);
            if (J->with_theta) W += g->dir * atan2(y - cy, x - cx) / (2 * M_PI);
            J->out[i] = (float)W;
        }
    }
}
int fit_set_grid_cpu(const fit_set *s, const double o[3], const int64_t n[3], double step, int with_theta, const uint8_t *mask, float *out) {
    if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0) return 0;
    grid_cpu_job J = {s, o, n, step, with_theta, mask, out};
    fit_zpar((int)n[0], fit_nthreads((int)n[0]), grid_cpu_rows, &J);
    return 0;
}

/* one field array (name: pc ps t0..t5 sp w) of a fit-field directory, max-pooled by pool per axis (raw or zarr; read in z-blocks) */
typedef struct { const uint8_t *blk; uint8_t *out; int z0, nz, H, W, Hp, Wp, pool; } ffa_pool_job;
static void ffa_pool_rows(void *ctx, int yp0, int yp1, int t) {   /* fit_field_array: the max over pool^3 into pooled rows [yp0, yp1) */
    const ffa_pool_job *J = ctx; const int P = J->pool, W = J->W, Wf = W / P * P; (void)t;
    uint8_t *acc = malloc((size_t)W);   /* the element-wise max of the block's rows of one pooled (z, y) cell row, then reduced along x */
    for (int a0 = 0; a0 < J->nz;) {
        const int zp = (J->z0 + a0) / P; int a1 = a0; while (a1 < J->nz && (J->z0 + a1) / P == zp) a1++;
        uint8_t *o = J->out + (size_t)zp * J->Hp * J->Wp;
        for (int yp = yp0; yp < yp1; yp++) {
            const int y1 = (yp + 1) * P < J->H ? (yp + 1) * P : J->H;
            memset(acc, 0, (size_t)W);
            for (int a = a0; a < a1; a++) for (int y = yp * P; y < y1; y++) {
                const uint8_t *r = J->blk + ((size_t)a * J->H + y) * W;
                for (int x = 0; x < W; x++) acc[x] = r[x] > acc[x] ? r[x] : acc[x];
            }
            uint8_t *d = o + (size_t)yp * J->Wp; int x = 0;
            for (; x < Wf; x += P) { uint8_t m = d[x / P]; for (int q = 0; q < P; q++) if (acc[x + q] > m) m = acc[x + q]; d[x / P] = m; }
            for (; x < W; x++) if (acc[x] > d[x / P]) d[x / P] = acc[x];
        }
        a0 = a1;
    }
    free(acc);
}
/* fit_field_array's pooled read chunk by chunk: the stored chunks (z3_chunk_map) on threads, each read alone (z3_read of its box part)
   and max-pooled straight into the output; absent chunks are the fill 0, the output's zeros. Every output cell lies in one chunk (chunk
   edges on pool multiples of the box), so the cells are the block path's maxima: the same bytes without the 1 GB block buffers, their
   fill memsets and copies. */
typedef struct { z3 *za; const int64_t *list; int64_t nlist; const int *chunk; const long long *zoff; int D, H, W, Hp, Wp, pool; uint8_t *out; _Atomic int64_t next; _Atomic int fail; } ffa_chunk_job;
static void *ffa_chunk_worker(void *arg) {
    ffa_chunk_job *J = arg; const int P = J->pool; const size_t cv = (size_t)J->chunk[0] * J->chunk[1] * J->chunk[2];
    uint8_t *buf = malloc(cv), *acc = malloc((size_t)J->chunk[2]); if (!buf || !acc) { atomic_store(&J->fail, 1); free(buf); free(acc); return nullptr; }
    for (int64_t t; (t = atomic_fetch_add(&J->next, 1)) < J->nlist && !atomic_load(&J->fail);) {
        const int64_t *c = J->list + 3 * t; int64_t o[3], n[3], lo[3];   /* lo: the part's box-local origin */
        const int64_t box[3] = {J->D, J->H, J->W};
        for (int d = 0; d < 3; d++) {
            const int64_t a = c[d] * J->chunk[d] - J->zoff[d], b = a + J->chunk[d];
            lo[d] = a < 0 ? 0 : a; const int64_t hi = b < box[d] ? b : box[d];
            n[d] = hi - lo[d]; o[d] = lo[d] + J->zoff[d];
        }
        if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0) continue;
        if (z3_read(J->za, o, n, buf, 1)) { atomic_store(&J->fail, 1); break; }
        const int nx = (int)n[2], nxf = nx / P * P;
        for (int64_t z0 = 0; z0 < n[0];) {
            const int64_t zp = (lo[0] + z0) / P; int64_t z1 = z0; while (z1 < n[0] && (lo[0] + z1) / P == zp) z1++;
            for (int64_t y0 = 0; y0 < n[1];) {
                const int64_t yp = (lo[1] + y0) / P; int64_t y1 = y0; while (y1 < n[1] && (lo[1] + y1) / P == yp) y1++;
                memset(acc, 0, (size_t)nx);
                for (int64_t z = z0; z < z1; z++) for (int64_t y = y0; y < y1; y++) {
                    const uint8_t *r = buf + ((size_t)z * n[1] + y) * nx;
                    for (int x = 0; x < nx; x++) acc[x] = r[x] > acc[x] ? r[x] : acc[x];
                }
                uint8_t *d = J->out + ((size_t)zp * J->Hp + yp) * J->Wp + lo[2] / P; int x = 0;
                for (; x < nxf; x += P) { uint8_t m = d[x / P]; for (int q = 0; q < P; q++) if (acc[x + q] > m) m = acc[x + q]; d[x / P] = m; }
                for (; x < nx; x++) if (acc[x] > d[x / P]) d[x / P] = acc[x];
                y0 = y1;
            }
            z0 = z1;
        }
    }
    free(buf); free(acc);
    return nullptr;
}
/* 1 done (out filled), 0 not applicable (the block path), -1 error */
static int ffa_chunks(z3 *za, const long long zoff[3], int D, int H, int W, int Hp, int Wp, int pool, int nth, uint8_t *out) {
    const z3_meta *m = z3_meta_of(za);
    if (pool < 2 || m->fill != 0 || m->esize != 1 || !m->sharded || ufsm_env_on("UFSM_FFA_BLOCKS")) return 0;
    for (int d = 0; d < 3; d++) if (m->chunk[d] % pool || zoff[d] % pool || zoff[d] < 0) return 0;
    int64_t g[3]; uint8_t *bits = nullptr; if (z3_chunk_map(za, g, &bits) < 0) { free(bits); return 0; }
    const long long box[3] = {D, H, W}; int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) { c0[d] = zoff[d] / m->chunk[d]; c1[d] = (zoff[d] + box[d] - 1) / m->chunk[d]; if (c1[d] > g[d] - 1) c1[d] = g[d] - 1; }
    int64_t nl = 0, cap = 1024; int64_t *list = malloc((size_t)cap * 3 * sizeof *list);
    for (int64_t cz = c0[0]; cz <= c1[0]; cz++) for (int64_t cy = c0[1]; cy <= c1[1]; cy++) for (int64_t cx = c0[2]; cx <= c1[2]; cx++) {
        if (!bits[(cz * g[1] + cy) * g[2] + cx]) continue;
        if (nl == cap) { cap *= 2; list = realloc(list, (size_t)cap * 3 * sizeof *list); }
        list[3 * nl] = cz; list[3 * nl + 1] = cy; list[3 * nl + 2] = cx; nl++;
    }
    free(bits);
    ffa_chunk_job J = {za, list, nl, m->chunk, zoff, D, H, W, Hp, Wp, pool, out, 0, 0};
    pthread_t th[64]; const int nt = nth < 1 ? 1 : nth > 64 ? 64 : nth; int k = 0;
    for (; k < nt; k++) if (pthread_create(&th[k], nullptr, ffa_chunk_worker, &J)) break;
    if (!k) ffa_chunk_worker(&J);
    for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    free(list);
    return atomic_load(&J.fail) ? -1 : 1;
}
uint8_t *fit_field_array(const char *dir, const char *name, int pool, int shape[3]) {
    fit_fields_buf g; if (fit_fields_load(dir, 0, 0, 0, &g)) return nullptr;
    const int D = g.shape[0], H = g.shape[1], W = g.shape[2], Dp = (D + pool - 1) / pool, Hp = (H + pool - 1) / pool, Wp = (W + pool - 1) / pool;
    char p[1400]; snprintf(p, sizeof p, "%s/fields.json", dir); json *j = read_json_file(p);
    const int zarr = j && !strcmp(json_str(json_get(j, "format"), "raw"), "zarr"); long long zoff[3] = {0, 0, 0}, rsh[3] = {D, H, W};
    if (j) fields_crop(j, zarr, (const int[3]){D, H, W}, zoff, rsh);
    coreline core = {0}; const int ismask = !strcmp(name, "w") || !strcmp(name, "pc") || !strcmp(name, "ps"), cside = j && ismask ? fields_core(j, &core) : 0;
    const long long gorg[3] = {g.origin[0], g.origin[1], g.origin[2]}; const double gum = g.um, gh = g.h;
    json_free(j); fit_fields_free(&g);
    z3 *za = nullptr; FILE *fr = nullptr;
    if (zarr) za = z3_open(store_open(dir), name, nullptr); else { snprintf(p, sizeof p, "%s/%s.raw", dir, name); fr = fopen(p, "rb"); }
    if (!za && !fr) return nullptr;
    /* blocks of >= 64 rows (whole pools): a one-row read decoded the 16-row codec blocks of its chunks for every row (~16x the decode);
       the reads (only the needed substreams), the fill and the pooling on --threads. A sharded zarr at a pool dividing its chunks goes
       chunk by chunk instead (ffa_chunks: PHerc0490B's w at pool 8, 80 G cells: 190 s -> 6 s, the same bytes) */
    const int zb = pool >= 64 ? pool : 64 / pool * pool, nth = fit_nthreads(1 << 20);
    uint8_t *out = calloc((size_t)Dp * Hp * Wp, 1);
    const int byc = out && za && !cside ? ffa_chunks(za, zoff, D, H, W, Hp, Wp, pool, nth, out) : 0;
    if (byc < 0) { free(out); out = nullptr; }
    uint8_t *blk = out && !byc ? malloc((size_t)zb * H * W) : nullptr;
    if (out && !byc && !blk) { free(out); out = nullptr; }
    for (int z0 = 0; z0 < (out && !byc ? D : 0); z0 += zb) {
        const int nz = z0 + zb <= D ? zb : D - z0;
        if (za) { const int64_t o[3] = {z0 + zoff[0], zoff[1], zoff[2]}, n[3] = {nz, H, W}; if (z3_read_sub(za, o, n, blk, nth, fit_submask(za, o[0], n[0]))) { free(out); out = nullptr; break; } }
        else { const long long o[3] = {z0 + zoff[0], zoff[1], zoff[2]}; const int64_t n[3] = {nz, H, W}; if (raw_read_box(fr, rsh, o, n, blk)) { free(out); out = nullptr; break; } }
        if (cside) core_mask(dir, &core, cside, gorg, gh, gum, D, z0, nz, H, W, name[0] == 'p' ? blk : nullptr, nullptr, name[0] == 'w' ? blk : nullptr);
        if (pool == 1) { memcpy(out + (size_t)z0 * H * W, blk, (size_t)nz * H * W); continue; }   /* (the max with the zeroed output) */
        ffa_pool_job PJ = {blk, out, z0, nz, H, W, Hp, Wp, pool};   /* pooled rows on threads (each output row by one thread: the same maxima) */
        fit_zpar(Hp, nth, ffa_pool_rows, &PJ);
    }
    free(blk); if (za) z3_close(za); if (fr) fclose(fr); if (cside) cl_free(&core);
    if (out) { shape[0] = Dp; shape[1] = Hp; shape[2] = Wp; }
    return out;
}

/* device bytes a fit set will take (fit_set_upload: its delta lattices; the axis / knot arrays are small): the delta.f32 sizes in DIR
   and its subdirectories (a set: global/, slabNNN/) */
size_t g_fit_dev_reserve;
size_t fit_set_disk_bytes(const char *dir) {
    size_t b = 0; struct stat st; char p[1400];
    snprintf(p, sizeof p, "%s/delta.f32", dir); if (!stat(p, &st)) b += (size_t)st.st_size;
    DIR *d = opendir(dir); if (!d) return b;
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s/delta.f32", dir, e->d_name); if (!stat(p, &st)) b += (size_t)st.st_size;
    }
    closedir(d); return b;
}
