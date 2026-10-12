/* Winding fitter host side (src/fit.h): label-free prior per z knot (pitch from the spacing field, refined by the phase
   coherence of r / pitch - dir theta / 2 pi; offset = its argument), then Adam on delta (coarse to fine) and the knots. */
#include "fit.h"
#include "nn.h"
#include "json.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

fit_cfg fit_cfg_default(void) {
    fit_cfg c = {0};
    c.dir = 0; c.knot_dz = 1024;
    c.nstage = 6; c.phase_stage = 4;
    const int lat[6] = {512, 256, 128, 64, 64, 32}, st[6] = {300, 300, 300, 300, 300, 300};
    memcpy(c.lattice, lat, sizeof lat); memcpy(c.steps, st, sizeof st);
    c.lr = 0.02f; c.lr_knot = 0.005f;
    c.w_phase = 1.f; c.w_normal = 1.f; c.w_density = 1.f; c.w_nondeg = 1.f; c.w_smooth = 0.1f; c.w_decay = 1e-4f;
    c.phase_ramp = 0.1f; c.core = 3.f;
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
double fit_W(const fit_model *m, double z, double y, double x) {
    const int na = (int)(m->D * m->h) + 1;
    const double cy = axis_lerp(m->cy, na, z), cx = axis_lerp(m->cx, na, z);
    const double r = hypot(y - cy, x - cx), th = atan2(y - cy, x - cx);
    const double U = r / knot_lerp(m->pitch, m->nk, m->knot_dz, z) + knot_lerp(m->offset, m->nk, m->knot_dz, z) +
                     lattice_at(m->delta, m->L, m->lz, m->ly, m->lx, z, y, x);
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
            const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0;
            if (mc * mc + ms * ms < 0.01) continue;
            const double cy = axis_lerp(f->cy, na, pz), cx = axis_lerp(f->cx, na, pz), py = (y + 0.5) * h, px = (x + 0.5) * h;
            v[ns++] = (fit_sample){hypot(py - cy, px - cx), atan2(py - cy, px - cx), f->w[i] / 255.0, mc + I * ms};
        }
        if (!nsp || !ns) { pk[k] = -1; continue; }
        qsort(sps, nsp, sizeof(float), cmp_f);
        const double p0 = sps[nsp / 2];
        double bestc = -1, bestp = p0;
        for (int s = 0; s <= 70; s++) { const double p = p0 * (0.7 + 0.01 * s), cc = sector_coherence(v, ns, p, 4 * p0); if (cc > bestc) { bestc = cc; bestp = p; } }
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
            const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0;
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

static float *prolong(const fit_model *m, int L, int lz, int ly, int lx) {   /* old delta at the new lattice nodes */
    float *d = calloc((size_t)lz * ly * lx, sizeof(float));
    if (m->delta) for (int a = 0; a < lz; a++) for (int b = 0; b < ly; b++) for (int e = 0; e < lx; e++)
        d[((size_t)a * ly + b) * lx + e] = (float)lattice_at(m->delta, m->L, m->lz, m->ly, m->lx, (double)a * L, (double)b * L, (double)e * L);
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

static fit_model *fit_run_dir(const fit_fields *f, const fit_cfg *c) {
    fit_model *m = calloc(1, sizeof *m);
    m->D = f->D; m->H = f->H; m->W = f->W; m->h = f->h;
    const int na = (int)(f->D * f->h) + 1;
    m->cy = malloc(na * sizeof(float)); m->cx = malloc(na * sizeof(float));
    memcpy(m->cy, f->cy, na * sizeof(float)); memcpy(m->cx, f->cx, na * sizeof(float));
    if (fit_init_prior(f, c, m)) { fit_free(m); return nullptr; }
    const size_t n = (size_t)f->D * f->H * f->W;
    size_t ndata = 0; for (size_t i = 0; i < n; i++) ndata += f->w[i] != 0;
    uint8_t *dpc = nn_malloc(n), *dps = nn_malloc(n), *dt = nn_malloc(6 * n), *dsp = nn_malloc(n), *dw = nn_malloc(n);
    nn_h2d(dpc, f->pc, n); nn_h2d(dps, f->ps, n); nn_h2d(dsp, f->sp, n); nn_h2d(dw, f->w, n);
    for (int k = 0; k < 6; k++) nn_h2d(dt + k * n, f->t[k], n);
    float *dcy = nn_malloc(na * 4), *dcx = nn_malloc(na * 4);
    nn_h2d(dcy, f->cy, na * 4); nn_h2d(dcx, f->cx, na * 4);
    const int nk = m->nk;
    float *dpi = nn_malloc(nk * 4), *dof = nn_malloc(nk * 4), *gpi = nn_malloc(nk * 4), *gof = nn_malloc(nk * 4);
    float *mpi = nn_malloc(nk * 4), *vpi = nn_malloc(nk * 4), *mof = nn_malloc(nk * 4), *vof = nn_malloc(nk * 4);
    nn_h2d(dpi, m->pitch, nk * 4); nn_h2d(dof, m->offset, nk * 4);
    nn_zero(mpi, nk * 4); nn_zero(vpi, nk * 4); nn_zero(mof, nk * 4); nn_zero(vof, nk * 4);
    double *dparts = (double *)nn_malloc(6 * sizeof(double));
    const double norm = ndata ? 1.0 / ndata : 1;
    int kstep = 0;
    for (int st = 0; st < c->nstage; st++) {
        const int L = c->lattice[st], lz = (int)ceil(f->D * f->h / L) + 1, ly = (int)ceil(f->H * f->h / L) + 1, lx = (int)ceil(f->W * f->h / L) + 1;
        const size_t nl = (size_t)lz * ly * lx;
        float *hd = prolong(m, L, lz, ly, lx);
        free(m->delta); m->delta = hd; m->L = L; m->lz = lz; m->ly = ly; m->lx = lx;
        if (st == c->phase_stage) {   /* the current knots too: W reads them */
            nn_d2h(m->pitch, dpi, nk * 4); nn_d2h(m->offset, dof, nk * 4);
            fit_unwrap_init(m, f, 0.3);
        }
        float *dd = nn_malloc(nl * 4), *gd = nn_malloc(nl * 4), *md = nn_malloc(nl * 4), *vd = nn_malloc(nl * 4), *lap = nn_malloc(nl * 4);
        nn_h2d(dd, hd, nl * 4); nn_zero(md, nl * 4); nn_zero(vd, nl * 4);
        fit_dev fd = {f->D, f->H, f->W, (float)f->h, dpc, dps, dt, dsp, dw, dcy, dcx, na, m->dir, nk, m->knot_dz, dpi, dof, L, lz, ly, lx, dd,
                      0, (float)(c->w_normal * norm), (float)(c->w_density * norm), (float)(c->w_nondeg * norm), c->core};
        const float ws = (float)(c->w_smooth / nl), wdec = (float)(c->w_decay / nl);
        for (int it = 0; it < c->steps[st]; it++) {
            const double ramp = st < c->phase_stage ? 0 : st == c->phase_stage ? c->phase_ramp + (1 - c->phase_ramp) * (double)it / (c->steps[st] > 1 ? c->steps[st] - 1 : 1) : 1;
            fd.w_phase = (float)(c->w_phase * ramp * norm);
            nn_zero(gd, nl * 4); nn_zero(gpi, nk * 4); nn_zero(gof, nk * 4); nn_zero(dparts, 6 * sizeof(double));
            nn_fit_loss(&fd, dparts, gd, gpi, gof);
            nn_fit_smooth(dd, lz, ly, lx, ws, wdec, lap, dparts, gd);
            nn_adamw(dd, gd, md, vd, nl, c->lr, 0.9f, 0.999f, 1e-8f, 0.f, it + 1);
            ++kstep;
            nn_adamw(dof, gof, mof, vof, nk, c->lr_knot, 0.9f, 0.999f, 1e-8f, 0.f, kstep);
            nn_adamw(dpi, gpi, mpi, vpi, nk, c->lr_knot * 20.f, 0.9f, 0.999f, 1e-8f, 0.f, kstep);
            if (c->verbose && (it == c->steps[st] - 1 || (it + 1) % 100 == 0)) {
                double p[6]; nn_sync(); nn_d2h(p, dparts, sizeof p);
                fprintf(stderr, "fit stage %d (lattice %d, %zu nodes) step %d: loss %.5f phase %.5f normal %.5f density %.5f nondeg %.5f smooth %.6f\n",
                        st, L, nl, it + 1, p[0] + p[5], p[1], p[2], p[3], p[4], p[5]);
            }
        }
        nn_sync(); nn_d2h(hd, dd, nl * 4);
        nn_d2h(m->loss, dparts, 6 * sizeof(double)); m->loss[0] += m->loss[5];
        nn_free(dd); nn_free(gd); nn_free(md); nn_free(vd); nn_free(lap);
    }
    nn_d2h(m->pitch, dpi, nk * 4); nn_d2h(m->offset, dof, nk * 4);
    nn_free(dpc); nn_free(dps); nn_free(dt); nn_free(dsp); nn_free(dw); nn_free(dcy); nn_free(dcx);
    nn_free(dpi); nn_free(dof); nn_free(gpi); nn_free(gof); nn_free(mpi); nn_free(vpi); nn_free(mof); nn_free(vof); nn_free(dparts);
    return m;
}

/* the direction from the prior's ring vote; when it is ambiguous (|median winding per ring| < 0.5 or too few rings) both
   directions are fitted and the lower final loss wins (a wrong direction cannot fit the phase around the circle) */
fit_model *fit_run(const fit_fields *f, const fit_cfg *c) {
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
    fprintf(f, "{\"version\": 1, \"dir\": %d, \"knot_dz\": %d, \"L\": %d, \"lattice\": [%d, %d, %d], \"shape\": [%d, %d, %d], \"h\": %.9g, "
               "\"origin\": [%lld, %lld, %lld], \"loss\": %.9g, \"pitch\": [", m->dir, m->knot_dz, m->L, m->lz, m->ly, m->lx, m->D, m->H, m->W, m->h,
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
    m->h = json_num(json_get(j, "h"), 8); m->loss[0] = json_num(json_get(j, "loss"), 0);
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

int fit_ray_root(const fit_model *m, double z, double theta, double k, double r_hint, double *r) {
    const int na = (int)(m->D * m->h) + 1;
    const double cy = axis_lerp(m->cy, na, z), cx = axis_lerp(m->cx, na, z), sy = sin(theta), sx = cos(theta);
    const double Y = m->H * m->h, X = m->W * m->h;
    double rlo = 0, rhi = 1e9;   /* the part of the ray inside the box */
    const double c0[2] = {cy, cx}, dv[2] = {sy, sx}, ext[2] = {Y, X};
    for (int d = 0; d < 2; d++) {
        if (fabs(dv[d]) < 1e-12) { if (c0[d] < 0 || c0[d] > ext[d]) return -1; continue; }
        double a = (0 - c0[d]) / dv[d], b = (ext[d] - c0[d]) / dv[d]; if (a > b) { double t = a; a = b; b = t; }
        if (a > rlo) rlo = a;
        if (b < rhi) rhi = b;
    }
    if (rhi <= rlo) return -1;
    const double pit = knot_lerp(m->pitch, m->nk, m->knot_dz, z), dr = pit / 4;
    double best = -1, bestd = 1e300, rp = rlo, fp = fit_W(m, z, cy + rlo * sy, cx + rlo * sx) - k;
    for (double rr = rlo + dr; rp < rhi; rr += dr) {
        if (rr > rhi) rr = rhi;
        const double fr = fit_W(m, z, cy + rr * sy, cx + rr * sx) - k;
        if ((fp < 0) != (fr < 0) && fabs(fp - fr) < 0.5) {   /* a crossing (not the theta cut of a nearby ray: |jump| < half a turn) */
            double a = rp, b = rr, fa = fp;
            for (int it = 0; it < 40; it++) { const double mid = 0.5 * (a + b), fm = fit_W(m, z, cy + mid * sy, cx + mid * sx) - k; if ((fa < 0) == (fm < 0)) { a = mid; fa = fm; } else b = mid; }
            const double root = 0.5 * (a + b);
            if (fabs(root - r_hint) < bestd) { bestd = fabs(root - r_hint); best = root; }
        }
        rp = rr; fp = fr;
        if (rr >= rhi) break;
    }
    if (best < 0) return -1;
    *r = best;
    return 0;
}

int fit_fields_save(const char *dir, const long long origin[3], const int shape[3], int h, uint8_t *const a[10], const float *cy, const float *cx, int na) {
    static const char *nm[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    char path[1400]; snprintf(path, sizeof path, "mkdir -p '%s'", dir); if (system(path)) return -1;
    const size_t nf = (size_t)shape[0] * shape[1] * shape[2];
    for (int k = 0; k < 10; k++) {
        snprintf(path, sizeof path, "%s/%s.raw", dir, nm[k]);
        FILE *f = fopen(path, "wb"); if (!f || fwrite(a[k], 1, nf, f) != nf) { if (f) fclose(f); return -1; } fclose(f);
    }
    snprintf(path, sizeof path, "%s/fields.json", dir);
    FILE *f = fopen(path, "w"); if (!f) return -1;
    fprintf(f, "{\"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"axis\": [", origin[0], origin[1], origin[2], shape[0], shape[1], shape[2], h);
    for (int z = 0; z < na; z++) fprintf(f, "%s[%.3f, %.3f]", z ? ", " : "", cy[z], cx[z]);
    fprintf(f, "]}\n");
    return fclose(f) ? -1 : 0;
}
