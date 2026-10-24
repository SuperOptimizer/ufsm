/* Scroll-damage augmentations of stored-phase windows: see scrollaug.h */
#include "scrollaug.h"
#include "band.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t s; } sa_rng;
static uint64_t sa_next(sa_rng *r) { uint64_t z = (r->s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double sa_unif(sa_rng *r) { return (double)(sa_next(r) >> 11) * 0x1.0p-53; }
static void sa_unit(sa_rng *r, double v[3]) {
    const double z = 2 * sa_unif(r) - 1, a = 2 * M_PI * sa_unif(r), q = sqrt(fmax(0, 1 - z * z));
    v[0] = z; v[1] = q * cos(a); v[2] = q * sin(a);
}
static double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static int norm3(double v[3]) { const double l = sqrt(dot3(v, v)); if (l < 1e-12) return 1; for (int d = 0; d < 3; d++) v[d] /= l; return 0; }

static inline int pdiff(int a, int b) {   /* a - b as a phase difference in codes, -126 .. 125 */
    int d = a - b; if (d >= PHASE_PERIOD / 2) d -= PHASE_PERIOD; if (d < -PHASE_PERIOD / 2) d += PHASE_PERIOD; return d;
}
int sa_phase_normal(const uint8_t *ph, int D, int H, int W, int inset, int stride, double n[3], double *spacing) {
    const int lo = inset < 1 ? 1 : inset, b0[3] = {lo, lo, lo}, b1[3] = {D - lo, H - lo, W - lo};
    return sa_phase_normal_box(ph, D, H, W, b0, b1, stride, n, spacing);
}
int sa_phase_normal_box(const uint8_t *ph, int D, int H, int W, const int lo3[3], const int hi3[3], int stride, double n[3], double *spacing) {
    if (stride < 1) stride = 1;
    double T[6] = {0}, gs[3] = {0, 0, 0};
    enum { NB = 1024 }; const double gmax = .25;   /* (|grad| histogram, turns per voxel) */
    uint64_t hist[NB] = {0}, cnt = 0;
    const size_t sy = (size_t)W, sz = (size_t)H * W;
    const int dim[3] = {D, H, W}; int lo[3], hi[3];
    for (int d = 0; d < 3; d++) { lo[d] = lo3[d] < 1 ? 1 : lo3[d]; hi[d] = hi3[d] > dim[d] - 1 ? dim[d] - 1 : hi3[d]; }
    for (int z = lo[0]; z < hi[0]; z += stride) for (int y = lo[1]; y < hi[1]; y += stride) for (int x = lo[2]; x < hi[2]; x += stride) {
        const size_t k = z * sz + y * sy + x; const uint8_t c = ph[k];
        if (c >= PHASE_PERIOD) continue;
        const uint8_t a0 = ph[k + sz], b0 = ph[k - sz], a1 = ph[k + sy], b1 = ph[k - sy], a2 = ph[k + 1], b2 = ph[k - 1];
        if (a0 >= PHASE_PERIOD || b0 >= PHASE_PERIOD || a1 >= PHASE_PERIOD || b1 >= PHASE_PERIOD || a2 >= PHASE_PERIOD || b2 >= PHASE_PERIOD) continue;
        const int d0 = pdiff(a0, b0), d1 = pdiff(a1, b1), d2 = pdiff(a2, b2);
        if (abs(d0) > PHASE_PERIOD / 4 || abs(d1) > PHASE_PERIOD / 4 || abs(d2) > PHASE_PERIOD / 4) continue;   /* (a label seam) */
        const double g[3] = {d0 / (2. * PHASE_PERIOD), d1 / (2. * PHASE_PERIOD), d2 / (2. * PHASE_PERIOD)};
        T[0] += g[0] * g[0]; T[1] += g[0] * g[1]; T[2] += g[0] * g[2]; T[3] += g[1] * g[1]; T[4] += g[1] * g[2]; T[5] += g[2] * g[2];
        for (int d = 0; d < 3; d++) gs[d] += g[d];
        const double m = sqrt(dot3(g, g)); if (m <= 0) continue;
        int b = (int)(m / gmax * NB); if (b >= NB) b = NB - 1;
        hist[b]++; cnt++;
    }
    if (cnt < 32) return (int)cnt;
    double v[3] = {.577, .577, .577};
    for (int it = 0; it < 64; it++) {
        const double w[3] = {T[0] * v[0] + T[1] * v[1] + T[2] * v[2], T[1] * v[0] + T[3] * v[1] + T[4] * v[2], T[2] * v[0] + T[4] * v[1] + T[5] * v[2]};
        memcpy(v, w, sizeof v); if (norm3(v)) return 0;
    }
    if (dot3(v, gs) < 0) for (int d = 0; d < 3; d++) v[d] = -v[d];   /* (oriented: the phase grows along n) */
    memcpy(n, v, sizeof v);
    uint64_t acc = 0; int b = 0; while (b < NB - 1 && (acc += hist[b]) * 2 < cnt) b++;
    *spacing = 1. / ((b + .5) * gmax / NB);
    return (int)cnt;
}

void sa_tear_make(sa_tear *t, uint64_t seed, int P, const double n[3], double spacing, double maxd) {
    memset(t, 0, sizeof *t);
    sa_rng r = {seed};
    double nn[3] = {n[0], n[1], n[2]}; if (norm3(nn)) { nn[0] = 1; nn[1] = nn[2] = 0; }
    /* the cut's normal: perpendicular to the sheets within 30 degrees (the cut crosses them) */
    double u[3];
    do { sa_unit(&r, u); const double a = dot3(u, nn); for (int d = 0; d < 3; d++) u[d] -= a * nn[d]; } while (norm3(u));
    const double tilt = (2 * sa_unif(&r) - 1) * M_PI / 6;
    for (int d = 0; d < 3; d++) t->c[d] = cos(tilt) * u[d] + sin(tilt) * nn[d];
    norm3(t->c);
    for (int d = 0; d < 3; d++) t->p0[d] = .5 * (P - 1) + (2 * sa_unif(&r) - 1) * P / 8.;
    for (int j = 0; j < 3; j++) {   /* the surface's waviness: wavelengths P / 2 .. 2 P, amplitudes up to P / 24 each */
        double kd[3]; sa_unit(&r, kd); const double lam = P * (.5 + 1.5 * sa_unif(&r));
        for (int d = 0; d < 3; d++) t->k[j][d] = 2 * M_PI / lam * kd[d];
        t->a[j] = sa_unif(&r) * P / 24.; t->f[j] = 2 * M_PI * sa_unif(&r);
    }
    /* the displacement: 0.3 .. 1.3 turns along the normal, half the time also -1 .. 1 turns along the sheet */
    const double an = (.3 + sa_unif(&r)) * (sa_unif(&r) < .5 ? -1 : 1);
    double at = 0, tv[3] = {0, 0, 0};
    if (sa_unif(&r) < .5) {
        at = 2 * sa_unif(&r) - 1;
        do { sa_unit(&r, tv); const double a = dot3(tv, nn); for (int d = 0; d < 3; d++) tv[d] -= a * nn[d]; } while (norm3(tv));
    }
    double D[3]; for (int d = 0; d < 3; d++) D[d] = (an * nn[d] + at * tv[d]) * spacing;
    const double L = sqrt(dot3(D, D)), sc = L > maxd && L > 0 ? maxd / L : 1;   /* (capped: the padded read's margin) */
    for (int d = 0; d < 3; d++) D[d] *= sc;
    t->turns_n = an * sc; t->turns_t = at * sc;
    for (int d = 0; d < 3; d++) { const int Di = (int)lrint(D[d]), h = (int)lrint(.5 * D[d]); t->dp[d] = h; t->dm[d] = h - Di; }
    const double s = sa_unif(&r);
    t->seam = s < 1. / 3 ? 0 : s < 2. / 3 ? 1 : 2;
    t->width = t->seam == 1 ? 1 + 2 * sa_unif(&r) : t->seam == 2 ? 2 + 4 * sa_unif(&r) : 0;
}
double sa_tear_h(const sa_tear *t, double z, double y, double x) {
    const double p[3] = {z, y, x};
    double h = t->c[0] * (z - t->p0[0]) + t->c[1] * (y - t->p0[1]) + t->c[2] * (x - t->p0[2]);
    for (int j = 0; j < 3; j++) h += t->a[j] * sin(dot3(t->k[j], p) + t->f[j]);
    return h;
}
int sa_tear_margin(const sa_tear *t) {
    int m = 0; for (int d = 0; d < 3; d++) { if (abs(t->dp[d]) > m) m = abs(t->dp[d]); if (abs(t->dm[d]) > m) m = abs(t->dm[d]); }
    return m;
}
void sa_tear_apply(const sa_tear *t, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, const uint8_t *w4_pad, const uint8_t *wc_pad,
                   uint8_t void_ct, uint8_t *ct, uint8_t *ph, uint8_t *w4, uint8_t *wc) {
    const int Pp = P + 2 * M; const size_t sy = (size_t)Pp, sz = (size_t)Pp * Pp;
    const double hw = .5 * t->width;
    double *h = malloc((size_t)P * sizeof *h);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        /* h along the row: the plane plus three sines advanced by rotation (one sin / cos per sine per row) */
        double cs[3], sn[3], dc[3], ds[3];
        for (int j = 0; j < 3; j++) {
            const double a0 = t->k[j][0] * z + t->k[j][1] * y + t->f[j]; cs[j] = cos(a0); sn[j] = sin(a0);
            dc[j] = cos(t->k[j][2]); ds[j] = sin(t->k[j][2]);
        }
        const double base = t->c[0] * (z - t->p0[0]) + t->c[1] * (y - t->p0[1]) - t->c[2] * t->p0[2];
        int np = 0;
        for (int x = 0; x < P; x++) {
            double v = base + t->c[2] * x;
            for (int j = 0; j < 3; j++) { v += t->a[j] * sn[j]; const double c2 = cs[j] * dc[j] - sn[j] * ds[j]; sn[j] = sn[j] * dc[j] + cs[j] * ds[j]; cs[j] = c2; }
            h[x] = v; np += v > 0;
        }
        const size_t ko = ((size_t)z * P + y) * P;
        const size_t rp = (size_t)(z - t->dp[0] + M) * sz + (size_t)(y - t->dp[1] + M) * sy + (size_t)(M - t->dp[2]);
        const size_t rm = (size_t)(z - t->dm[0] + M) * sz + (size_t)(y - t->dm[1] + M) * sy + (size_t)(M - t->dm[2]);
        for (int x = 0; x < P; x++) {
            const int pos = h[x] > 0; const size_t s = (pos ? rp : rm) + x;
            uint8_t cv = ct_pad[s], pv = ph_pad[s], wv = w4_pad ? w4_pad[s] : 255, cc = wc_pad ? wc_pad[s] : 255;
            if (t->seam && fabs(h[x]) < hw) {
                if (t->seam == 1) cv = void_ct;
                else {
                    const uint8_t a = ct_pad[rm + x], b = ct_pad[rp + x]; const double wgt = (h[x] + hw) / (2 * hw);
                    cv = a && b ? (uint8_t)lrint((1 - wgt) * a + wgt * b) : 0;
                }
                pv = PHASE_UNKNOWN; wv = cc = 255;
            }
            ct[ko + x] = cv; ph[ko + x] = pv;
            if (w4) w4[ko + x] = wv;
            if (wc) wc[ko + x] = cc;
        }
    }
    free(h);
}
uint8_t sa_ct_quantile(const uint8_t *ct, size_t n, double q) {
    uint64_t hist[256] = {0}, tot = 0;
    for (size_t k = 0; k < n; k++) hist[ct[k]]++;
    tot = n - hist[0]; if (!tot) return 0;
    uint64_t acc = 0; int v = 1; while (v < 255 && (double)(acc += hist[v]) < q * (double)tot) v++;
    return (uint8_t)v;
}

/* ---------------- --aug-delam: peels and holes ---------------- */
int sa_delam_margin(double spacing) { (void)spacing; return 6; }   /* (g0 <= 4 voxels) */
static void sa_perp(const double n[3], double t1[3], double t2[3]) {
    const double a[3] = {fabs(n[0]) < .9 ? 1 : 0, fabs(n[0]) < .9 ? 0 : 1, 0};
    t1[0] = a[1] * n[2] - a[2] * n[1]; t1[1] = a[2] * n[0] - a[0] * n[2]; t1[2] = a[0] * n[1] - a[1] * n[0]; norm3(t1);
    t2[0] = n[1] * t1[2] - n[2] * t1[1]; t2[1] = n[2] * t1[0] - n[0] * t1[2]; t2[2] = n[0] * t1[1] - n[1] * t1[0];
}
int sa_delam_make(sa_delam *d, uint64_t seed, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, double spacing, double p_hole, uint8_t iso) {
    memset(d, 0, sizeof *d);
    sa_rng r = {seed}; const int Pp = P + 2 * M; const size_t sy = (size_t)Pp, sz = (size_t)Pp * Pp;
    d->spacing = spacing; d->iso = iso;
    const int want = 1 + (int)(sa_unif(&r) * 3);
    for (int tries = 0; tries < 400 && d->n < want; tries++) {
        int c[3]; for (int k = 0; k < 3; k++) c[k] = P / 8 + (int)(sa_unif(&r) * (P * 3 / 4));
        const size_t kc = (size_t)(c[0] + M) * sz + (size_t)(c[1] + M) * sy + (size_t)(c[2] + M);
        if (ct_pad[kc] < iso || ph_pad[kc] >= PHASE_PERIOD) continue;
        const int h = (int)ceil(2 * spacing), lo[3] = {c[0] + M - h, c[1] + M - h, c[2] + M - h}, hi[3] = {c[0] + M + h, c[1] + M + h, c[2] + M + h};
        sa_site *s = &d->s[d->n]; double sp;
        if (sa_phase_normal_box(ph_pad, Pp, Pp, Pp, lo, hi, 1 + h / 8, s->n, &sp) < 32) continue;
        for (int k = 0; k < 3; k++) s->c[k] = c[k];
        sa_perp(s->n, s->t1, s->t2);
        s->kind = sa_unif(&r) < p_hole; s->phc = ph_pad[kc]; s->half = .4 * spacing;
        if (s->kind == 0) { s->R = spacing * (1 + 3 * sa_unif(&r)); s->R2 = s->R; s->g0 = 1 + sa_unif(&r) * (fmin(4, .25 * spacing) - 1); if (s->g0 < 1) s->g0 = 1; }
        else {
            double cs = 0, sn = 0;   /* (the hole's sheet centre: the CT-weighted circular mean phase of the papyrus on the normal through c, +-.4 turn) */
            for (int k = -(int)(.4 * spacing); k <= (int)(.4 * spacing); k++) {
                int q[3]; for (int d = 0; d < 3; d++) { q[d] = c[d] + M + (int)lrint(k * s->n[d]); if (q[d] < 0) q[d] = 0; if (q[d] > Pp - 1) q[d] = Pp - 1; }
                const size_t kq = (size_t)q[0] * sz + (size_t)q[1] * sy + (size_t)q[2];
                if (ct_pad[kq] < iso || ph_pad[kq] >= PHASE_PERIOD) continue;
                const double th = 2 * M_PI * ph_pad[kq] / PHASE_PERIOD, wq = ct_pad[kq] - iso + 1;
                cs += wq * cos(th); sn += wq * sin(th);
            }
            if (cs != 0 || sn != 0) { const int pc = (int)lrint(atan2(sn, cs) / (2 * M_PI) * PHASE_PERIOD); s->phc = (uint8_t)((pc % PHASE_PERIOD + PHASE_PERIOD) % PHASE_PERIOD); }
            s->R = spacing * (.5 + 2 * sa_unif(&r)); s->R2 = spacing * (.5 + 2 * sa_unif(&r)); s->ash = .2 * sa_unif(&r); }
        for (int j = 0; j < 3; j++) { s->wob[j] = sa_unif(&r); s->wph[j] = 2 * M_PI * sa_unif(&r); }
        d->n++;
    }
    d->void_ct = sa_ct_quantile(ct_pad, (size_t)Pp * Pp * Pp, .05);
    return d->n;
}
static double tri_u8(const uint8_t *v, int Pp, const double q[3]) {   /* trilinear, clamped to the box */
    double f[3]; int i[3];
    for (int d = 0; d < 3; d++) { double x = q[d]; if (x < 0) x = 0; if (x > Pp - 1.001) x = Pp - 1.001; i[d] = (int)x; f[d] = x - i[d]; }
    const size_t sy = (size_t)Pp, sz = (size_t)Pp * Pp, k = i[0] * sz + i[1] * sy + i[2];
    const double c00 = v[k] + f[2] * (v[k + 1] - v[k]), c01 = v[k + sy] + f[2] * (v[k + sy + 1] - v[k + sy]);
    const double c10 = v[k + sz] + f[2] * (v[k + sz + 1] - v[k + sz]), c11 = v[k + sz + sy] + f[2] * (v[k + sz + sy + 1] - v[k + sz + sy]);
    const double c0 = c00 + f[1] * (c01 - c00), c1 = c10 + f[1] * (c11 - c10);
    return c0 + f[0] * (c1 - c0);
}
void sa_delam_apply(const sa_delam *dl, double t, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, uint8_t *ct, uint8_t *ph) {
    const int Pp = P + 2 * M; const size_t sy = (size_t)Pp, sz = (size_t)Pp * Pp;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        const size_t s = (size_t)(z + M) * sz + (size_t)(y + M) * sy + M, o = ((size_t)z * P + y) * P;
        memcpy(ct + o, ct_pad + s, (size_t)P); memcpy(ph + o, ph_pad + s, (size_t)P);
    }
    if (t <= 0) return;
    const double sp = dl->spacing, x2 = .5 * sp;
    for (int i = 0; i < dl->n; i++) {
        const sa_site *S = &dl->s[i];
        const double ext = fmax(S->R, S->R2) * 1.3 + sp + 2;
        int lo[3], hi[3];
        for (int d = 0; d < 3; d++) { lo[d] = (int)floor(S->c[d] - ext); hi[d] = (int)ceil(S->c[d] + ext); if (lo[d] < 0) lo[d] = 0; if (hi[d] > P) hi[d] = P; }
        for (int z = lo[0]; z < hi[0]; z++) for (int y = lo[1]; y < hi[1]; y++) for (int x = lo[2]; x < hi[2]; x++) {
            const double dv[3] = {z - S->c[0], y - S->c[1], x - S->c[2]}, a = dot3(dv, S->n);
            const size_t ki = (size_t)(z + M) * sz + (size_t)(y + M) * sy + (size_t)(x + M), ko = ((size_t)z * P + y) * P + x;
            if (S->kind == 0) {
                if (fabs(a) > .6 * sp) continue;
                const double tv[3] = {dv[0] - a * S->n[0], dv[1] - a * S->n[1], dv[2] - a * S->n[2]}, r2 = dot3(tv, tv) / (S->R * S->R);
                if (r2 >= 1) continue;
                const double fa = fabs(a), ta = fa < .4 * sp ? 1 : (.6 * sp - fa) / (.2 * sp);   /* (tapered off the plane: no step at |a| = .6) */
                const double b = t * S->g0 * (1 - r2) * (1 - r2) * ta;
                if (b < .05) continue;
                const uint8_t pv = ph_pad[ki];
                const double dl_ = pv < PHASE_PERIOD ? pdiff(pv, S->phc) / (double)PHASE_PERIOD * sp : a;   /* (signed distance past the split) */
                if (dl_ < 0) continue;
                if (dl_ < b) { if (ct[ko]) { ct[ko] = dl->void_ct; ph[ko] = pv < PHASE_PERIOD ? S->phc : PHASE_UNKNOWN; } continue; }   /* the opened gap (unknown phase stays unknown) */
                const double w = dl_ < x2 ? (x2 - dl_) / (x2 - b) : 0;   /* (linear from the gap edge to 0 at half a turn, before the phase wraps: the phase map stays monotone, slope x2 / (x2 - b)) */
                if (w <= 0) continue;
                const double q[3] = {z + M - S->n[0] * b * w, y + M - S->n[1] * b * w, x + M - S->n[2] * b * w};
                ct[ko] = (uint8_t)lrint(tri_u8(ct_pad, Pp, q));
                if (pv < PHASE_PERIOD) { int nv = (int)pv - (int)lrint(b * w * PHASE_PERIOD / sp); nv %= PHASE_PERIOD; if (nv < 0) nv += PHASE_PERIOD; ph[ko] = (uint8_t)nv; }   /* the phase remapped as a function of itself (no resampling jitter); an unknown voxel stays unknown */
            } else {
                if (fabs(a) > S->half) continue;
                const uint8_t pv = ph_pad[ki];
                if (pv >= PHASE_PERIOD) continue;
                const double pd = abs(pdiff(pv, S->phc)) / (double)PHASE_PERIOD;   /* (the centre's sheet only: within .22 turn, rim from .17) */
                if (pd >= .22) continue;
                const double u1 = dot3(dv, S->t1) / S->R, u2 = dot3(dv, S->t2) / S->R2, rho = sqrt(u1 * u1 + u2 * u2), th = atan2(u2, u1);
                double rag = 1; for (int j = 0; j < 3; j++) rag += .25 * S->wob[j] * sin((j + 2) * th + S->wph[j]);
                const double mr = (t * rag - rho) / .15;   /* (a soft rim .15 radius wide) */
                if (mr <= 0) continue;
                const double m = fmin(1, mr) * fmin(1, (.22 - pd) / .05), cv = ct_pad[ki];
                if (cv <= dl->void_ct || !ct[ko]) continue;
                const uint32_t hsh = (uint32_t)(((uint64_t)z * 73856093u) ^ ((uint64_t)y * 19349663u) ^ ((uint64_t)x * 83492791u)) * 2654435761u;
                const double spk = .7 + .6 * (hsh >> 8) / (double)(1u << 24), tg = dl->void_ct + S->ash * spk * (cv - dl->void_ct), v = cv + m * (tg - cv);   /* (ash keeps some fibre texture) */
                ct[ko] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lrint(v));
            }
        }
    }
}
