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
int sa_delam_margin(double spacing) { (void)spacing; return 8; }   /* (g0 <= 6 voxels) */
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
        if (s->kind == 0) {   /* the phase -> distance profile past the split on the normal through c: the stored phase is not linear in
                                 space (papyrus spans about half a turn over a few voxels, the void the rest) */
            for (int sd = 0; sd < 2; sd++) {   /* (sd 0 ahead along n, 1 behind) */
                float *T = s->T[sd]; const double sg = sd ? -1 : 1;
                int nc = 1; double tp = 0, op = 0; T[0] = 0;
                for (double tt = .25; tt <= 1.2 * spacing && nc < 127; tt += .25) {
                    int q[3]; for (int d = 0; d < 3; d++) { q[d] = c[d] + M + (int)lrint(sg * tt * s->n[d]); if (q[d] < 0) q[d] = 0; if (q[d] > Pp - 1) q[d] = Pp - 1; }
                    const uint8_t pq = ph_pad[(size_t)q[0] * sz + (size_t)q[1] * sy + (size_t)q[2]];
                    if (pq >= PHASE_PERIOD) continue;
                    const int off = ((sd ? s->phc - (int)pq : (int)pq - s->phc) + PHASE_PERIOD) % PHASE_PERIOD;
                    if (off > 200 || off <= op) continue;   /* (the other side of the split, or not past the furthest offset yet) */
                    for (; nc < 127 && nc <= off; nc++) T[nc] = (float)(tp + (nc - op) / (off - op) * (tt - tp));
                    tp = tt; op = off;
                }
                for (; nc < 127; nc++) T[nc] = T[nc - 1] + (float)(spacing / PHASE_PERIOD);   /* (not reached: the mean slope) */
                for (int k = 1; k < 127; k++) if (T[k] < T[k - 1] + 1e-3f) T[k] = T[k - 1] + 1e-3f;
            }
            s->R = spacing * (1 + 3 * sa_unif(&r)); s->R2 = s->R; s->g0 = 2 + sa_unif(&r) * (fmin(6, .4 * spacing) - 2); if (s->g0 < 1) s->g0 = 1; }
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
    const double sp = dl->spacing;
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
                /* the two halves move apart, each by up to b / 2 within its own half turn: side 0 ahead of the split (phase offsets 0 .. 126,
                   along n), side 1 behind (offsets 126 .. 252, against n); unknown phase: the side and distance from the plane */
                const int of = pv < PHASE_PERIOD ? (pv - S->phc + PHASE_PERIOD) % PHASE_PERIOD : -1;
                const int side = of >= 0 ? of > 126 : a < 0, off = of < 0 ? -1 : side ? PHASE_PERIOD - of : of;
                const float *T = S->T[side]; const double sg = side ? -1 : 1;
                const double d2 = T[126], hb = fmin(.5 * b, .5 * d2);   /* (the half-turn point; each half moves at most half way there: the phase map stretches <= 2x) */
                const double dl_ = off >= 0 ? T[off] : fabs(a);   /* (geometric distance from the split: the site's profile) */
                if (dl_ < hb) { if (ct[ko]) { ct[ko] = dl->void_ct; ph[ko] = off >= 0 ? S->phc : PHASE_UNKNOWN; } continue; }   /* the opened gap (unknown phase stays unknown) */
                const double w = dl_ < d2 ? (d2 - dl_) / (d2 - hb) : 0;   /* (linear from the gap edge to 0 at the half-turn point) */
                if (w <= 0) continue;
                const double q[3] = {z + M - sg * S->n[0] * hb * w, y + M - sg * S->n[1] * hb * w, x + M - sg * S->n[2] * hb * w};
                ct[ko] = (uint8_t)lrint(tri_u8(ct_pad, Pp, q));
                if (off >= 0) {   /* the phase where the material came from: the profile inverted at dl - hb w (monotone in the phase: no turn gained) */
                    const double ds = dl_ - hb * w; int lo = 0, hi = 126;
                    while (hi - lo > 1) { const int md = (lo + hi) / 2; if (T[md] <= ds) lo = md; else hi = md; }
                    double cf = lo + (ds - T[lo]) / (T[hi] - T[lo]); cf = cf < 0 ? 0 : cf > 126 ? 126 : cf;
                    ph[ko] = (uint8_t)((S->phc + (int)(sg * lrint(cf)) + PHASE_PERIOD) % PHASE_PERIOD);
                }
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

/* ---------------- --aug-sheet-density: per-wrap brightness, slow carbonisation field ---------------- */
static double sa_hash1(uint64_t seed, int64_t j) { sa_rng r = {seed ^ ((uint64_t)j * 0xd1342543de82ef95ull)}; sa_next(&r); return 2 * sa_unif(&r) - 1; }
int sa_density_make(sa_density *d, uint64_t seed, const uint8_t *ct, const uint8_t *ph, int D, int H, int W, uint8_t iso) {
    memset(d, 0, sizeof *d);
    const int mn = D < H ? (D < W ? D : W) : (H < W ? H : W), st = mn / 32 > 1 ? mn / 32 : 1, st4 = mn / 96 > 1 ? mn / 96 : 1;   /* (st4: the papyrus statistics on ~96^3 samples) */
    if (sa_phase_normal(ph, D, H, W, 0, st, d->n, &d->spacing) < 32 || !(d->spacing > 2)) return 0;
    const size_t N = (size_t)D * H * W;
    d->void_ct = sa_ct_quantile(ct, N, .05); d->iso = iso > d->void_ct ? iso : (uint8_t)(d->void_ct + 1);
    double cs = 0, sn = 0;   /* the papyrus' mean phase */
    for (int z = 0; z < D; z += st4) for (int y = 0; y < H; y += st4) for (int x = 0; x < W; x += st4) {
        const size_t k = ((size_t)z * H + y) * W + x;
        if (ct[k] < d->iso || ph[k] >= PHASE_PERIOD) continue;
        const double th = 2 * M_PI * ph[k] / PHASE_PERIOD; cs += cos(th); sn += sin(th);
    }
    if (cs == 0 && sn == 0) return 0;
    d->phc = fmod(atan2(sn, cs) / (2 * M_PI) * PHASE_PERIOD + PHASE_PERIOD, PHASE_PERIOD);
    double c2 = 0, s2 = 0;   /* c0: the circular mean of a / spacing - psi over the papyrus (the sheets then sit near integers) */
    for (int z = 0; z < D; z += st4) for (int y = 0; y < H; y += st4) for (int x = 0; x < W; x += st4) {
        const size_t k = ((size_t)z * H + y) * W + x;
        if (ct[k] < d->iso || ph[k] >= PHASE_PERIOD) continue;
        double ps = (ph[k] - d->phc) / PHASE_PERIOD; ps -= floor(ps + .5);
        const double kc = ((z - .5 * D) * d->n[0] + (y - .5 * H) * d->n[1] + (x - .5 * W) * d->n[2]) / d->spacing - ps;
        c2 += cos(2 * M_PI * kc); s2 += sin(2 * M_PI * kc);
    }
    d->c0 = atan2(s2, c2) / (2 * M_PI);
    sa_rng r = {seed}; d->seed = sa_next(&r);
    d->s_gain = .1 + .25 * sa_unif(&r); d->s_off = .15 * sa_unif(&r); d->s_slow = .3 * sa_unif(&r);
    const int mx = D > H ? (D > W ? D : W) : (H > W ? H : W);
    for (int j = 0; j < 3; j++) {
        double u[3]; sa_unit(&r, u); const double lam = (.7 + 1.8 * sa_unif(&r)) * mx;
        for (int a = 0; a < 3; a++) d->k[j][a] = 2 * M_PI / lam * u[a];
        d->f[j] = 2 * M_PI * sa_unif(&r);
    }
    return 1;
}
void sa_density_apply(const sa_density *d, double t, const uint8_t *ct, const uint8_t *ph, int D, int H, int W, uint8_t *out) {
    const size_t N = (size_t)D * H * W;
    if (t <= 0) { if (out != ct) memcpy(out, ct, N); return; }
    float psi[256], mt[256];
    for (int v = 0; v < 256; v++) { double ps = (v - d->phc) / PHASE_PERIOD; ps -= floor(ps + .5); psi[v] = (float)ps; }
    const double vd = d->void_ct, span = d->iso - vd, sp = d->spacing, gs = t * d->s_gain, os = t * d->s_off * span, ss = t * d->s_slow / 3;
    for (int v = 0; v < 256; v++) { double m = (v - vd) / span; mt[v] = (float)(m < 0 ? 0 : m > 1 ? 1 : m); }
    /* the per-wrap values for every integer kc can reach in this box (|a| / spacing + psi + c0) */
    const double amax = .5 * sqrt((double)D * D + (double)H * H + (double)W * W) / sp;
    const int jlo = (int)floor(-amax - 2), nj = (int)ceil(2 * amax + 5);
    float *kc = malloc((size_t)W * sizeof *kc), *sl = malloc((size_t)(W + 16) * sizeof *sl), *gv = malloc((size_t)nj * sizeof *gv), *ov = malloc((size_t)nj * sizeof *ov);
    int *kn = malloc((size_t)W * sizeof *kn);
    if (!kc || !kn || !gv || !ov || !sl) { free(kc); free(kn); free(gv); free(ov); free(sl); if (out != ct) memcpy(out, ct, N); return; }
    const uint64_t so = d->seed ^ 0x5bd1e995a5a5a5a5ull;
    for (int j = 0; j < nj; j++) { gv[j] = (float)(gs * sa_hash1(d->seed, jlo + j)); ov[j] = (float)(os * sa_hash1(so, jlo + j)); }
    /* the slow field (wavelengths >= .7 box) every 16 voxels along x, linear between */
    double cr[3], sr[3];
    for (int j = 0; j < 3; j++) { cr[j] = cos(16 * d->k[j][2]); sr[j] = sin(16 * d->k[j][2]); }
    const float vdf = (float)vd;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) {
        const size_t r0 = ((size_t)z * H + y) * W;
        const uint8_t *pr = ph + r0, *cv = ct + r0; uint8_t *orow = out + r0;
        const double a0 = ((z - .5 * D) * d->n[0] + (y - .5 * H) * d->n[1] - .5 * W * d->n[2]) / sp - d->c0, ax = d->n[2] / sp;
        for (int x = 0; x < W; x++) { kn[x] = pr[x] < PHASE_PERIOD; kc[x] = (float)(a0 + ax * x) - (kn[x] ? psi[pr[x]] : 0); }
        /* unknown phase: the nearest known kc along the row within 6 voxels (one pass each way; kn 2 + the distance of a forward fill) */
        { int last = -100; for (int x = 0; x < W; x++) { if (kn[x] == 1) { last = x; continue; } if (x - last <= 6) { kc[x] = kc[last]; kn[x] = 2 + (x - last); } } }
        { int last = W + 100; for (int x = W - 1; x >= 0; x--) { if (kn[x] == 1) { last = x; continue; } if (last - x <= 6 && (kn[x] == 0 || last - x < kn[x] - 2)) kc[x] = kc[last]; } }
        double c[3], s[3];
        for (int j = 0; j < 3; j++) { const double b = d->k[j][0] * z + d->k[j][1] * y + d->f[j]; c[j] = cos(b); s[j] = sin(b); }
        for (int x = 0; x <= W; x += 16) {
            sl[x] = (float)(1 + ss * (c[0] + c[1] + c[2]));
            for (int j = 0; j < 3; j++) { const double cn = c[j] * cr[j] - s[j] * sr[j]; s[j] = s[j] * cr[j] + c[j] * sr[j]; c[j] = cn; }
        }
        for (int x = 0; x < W; x++) {
            const uint8_t v = cv[x];
            const int x0 = x & ~15; const float sv = sl[x0] + (sl[x0 + 16 <= W ? x0 + 16 : x0] - sl[x0]) * (float)(x - x0) * (1.f / 16);
            const float fj = floorf(kc[x]), u = kc[x] - fj; float w = (u - .2f) * (1 / .6f); w = w < 0 ? 0 : w > 1 ? 1 : w * w * (3 - 2 * w);
            int j = (int)fj - jlo; j = j < 0 ? 0 : j > nj - 2 ? nj - 2 : j;   /* (wrap values flat within +-.2 of an integer, smooth between) */
            const float g = 1 + gv[j] + w * (gv[j + 1] - gv[j]), of = ov[j] + w * (ov[j + 1] - ov[j]);
            float o = vdf + ((float)v - vdf) * g * sv + of * mt[v];
            o = o < 1 ? 1 : o > 255 ? 255 : o;
            orow[x] = v ? (uint8_t)(o + .5f) : 0;
        }
    }
    free(kc); free(kn); free(gv); free(ov); free(sl);
}
