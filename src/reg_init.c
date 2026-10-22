/* ufsm reg init: a from-scratch affine between two scans of one scroll (no published transform).
   Both volumes at ~W um (pyramid level + in-memory 2x pooling); the scroll axis of each from per-slice mask centroids (weighted line fit);
   both resampled in cylinder coordinates (r, phi, h) about their own axis; the rotation about the axis and the offset along it from a
   masked NCC (Padfield) over (phi, h) shifts, accumulated over all rings r in the Fourier domain, for the 4 orientations (h and phi each
   kept or reversed). The best peak (sub-bin by parabola) gives the affine fine L0 xyz -> coarse L0 xyz written as a reg map JSON;
   refine it afterwards with reg boxes / reg fit --affine-only from coarse to fine levels. */
#include "reg.h"
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---------- FFT (radix 2, in place) ---------- */
static void fft1(double complex *a, int n, int inv) {
    for (int i = 1, j = 0; i < n; i++) { int b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { const double complex t = a[i]; a[i] = a[j]; a[j] = t; } }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = 2 * M_PI / len * (inv ? 1 : -1); const double complex wl = cos(ang) + I * sin(ang);
        for (int i = 0; i < n; i += len) { double complex w = 1; for (int k = 0; k < len / 2; k++) { const double complex u = a[i + k], v = a[i + k + len / 2] * w; a[i + k] = u + v; a[i + k + len / 2] = u - v; w *= wl; } }
    }
}
/* 2D [n0][n1] (row-major), forward or inverse (unnormalised) */
static void fft2(double complex *a, int n0, int n1, int inv, double complex *col) {
    for (int i = 0; i < n0; i++) fft1(a + (size_t)i * n1, n1, inv);
    for (int j = 0; j < n1; j++) { for (int i = 0; i < n0; i++) col[i] = a[(size_t)i * n1 + j]; fft1(col, n0, inv); for (int i = 0; i < n0; i++) a[(size_t)i * n1 + j] = col[i]; }
}

/* ---------- a volume in memory at ~W um ---------- */
typedef struct { uint8_t *v; int64_t n[3]; double f, um0; double a0[3], d[3], e1[3], e2[3]; double hlo, hhi, rmax; } mvol;   /* f: L0 voxels per voxel */
static int mvol_load(rvol *rv, double W, int nt, mvol *m) {
    const int l = rvol_level_for(rv, W); const int64_t *s = rv->shape[l];
    if ((double)s[0] * s[1] * s[2] > 2.5e9) { fprintf(stderr, "reg init: level %d of %s too large\n", l, rv->name); return -1; }
    uint8_t *v = malloc((size_t)s[0] * s[1] * s[2]); const int64_t o[3] = {0, 0, 0};
    if (!v || rvol_read_nt(rv, l, o, s, v, nt)) { free(v); return -1; }
    m->v = v; memcpy(m->n, s, sizeof m->n); m->f = (double)(1 << l); m->um0 = rv->um;
    while (m->um0 * m->f * 2 <= W * 1.05) {   /* pool 2x in memory (mean of the non-zero voxels; zero if all zero) */
        int64_t n2[3]; for (int a = 0; a < 3; a++) n2[a] = (m->n[a] + 1) / 2;
        uint8_t *w = malloc((size_t)n2[0] * n2[1] * n2[2]);
        for (int64_t z = 0; z < n2[0]; z++) for (int64_t y = 0; y < n2[1]; y++) for (int64_t x = 0; x < n2[2]; x++) {
            int sum = 0, c = 0;
            for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                const int64_t zz = 2 * z + dz, yy = 2 * y + dy, xx = 2 * x + dx;
                if (zz < m->n[0] && yy < m->n[1] && xx < m->n[2]) { const int t = m->v[((size_t)zz * m->n[1] + yy) * m->n[2] + xx]; if (t) { sum += t; c++; } }
            }
            w[((size_t)z * n2[1] + y) * n2[2] + x] = c >= 4 ? (uint8_t)((sum + c / 2) / c) : 0;
        }
        free(m->v); m->v = w; memcpy(m->n, n2, sizeof n2); m->f *= 2;
    }
    fprintf(stderr, "reg init: %s at %.1f um: %lld x %lld x %lld\n", rv->name, m->um0 * m->f, (long long)m->n[0], (long long)m->n[1], (long long)m->n[2]);
    return 0;
}
/* level voxel coords -> physical um (L0 voxel * um0) and back */
static inline double l2p(const mvol *m, double x) { return ((x + 0.5) * m->f - 0.5) * m->um0; }
static inline double p2l(const mvol *m, double p) { return (p / m->um0 + 0.5) / m->f - 0.5; }
static void cross(const double a[3], const double b[3], double c[3]) { c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0]; }
static void norm3(double a[3]) { const double n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int i = 0; i < 3; i++) a[i] /= n; }

static void mvol_axis(mvol *m) {
    const int64_t nz = m->n[0], ny = m->n[1], nx = m->n[2];
    double *A = calloc(nz, sizeof *A), *cx = calloc(nz, sizeof *cx), *cy = calloc(nz, sizeof *cy); double amax = 0;
    for (int64_t z = 0; z < nz; z++) {
        double s = 0, sx = 0, sy = 0;
        for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++) if (m->v[((size_t)z * ny + y) * nx + x]) { s++; sx += x; sy += y; }
        A[z] = s; if (s) { cx[z] = sx / s; cy[z] = sy / s; } amax = fmax(amax, s);
    }
    /* weighted line fits x(z), y(z) over slices with >= 30% of the largest area */
    double S = 0, Sz = 0, Szz = 0, Sx = 0, Szx = 0, Sy = 0, Szy = 0; int64_t zl = -1, zh = -1;
    for (int64_t z = 0; z < nz; z++) if (A[z] >= 0.3 * amax) {
        const double w = A[z]; S += w; Sz += w * z; Szz += w * z * z; Sx += w * cx[z]; Szx += w * z * cx[z]; Sy += w * cy[z]; Szy += w * z * cy[z];
        if (zl < 0) zl = z;
        zh = z;
    }
    const double den = S * Szz - Sz * Sz;
    const double bx = den > 0 ? (S * Szx - Sz * Sx) / den : 0, by = den > 0 ? (S * Szy - Sz * Sy) / den : 0;
    const double zm = Sz / S, xm = Sx / S, ym = Sy / S;
    m->a0[0] = l2p(m, xm); m->a0[1] = l2p(m, ym); m->a0[2] = l2p(m, zm);
    m->d[0] = bx; m->d[1] = by; m->d[2] = 1; norm3(m->d);
    const double xh[3] = {1, 0, 0}; const double dd = m->d[0];
    for (int i = 0; i < 3; i++) m->e1[i] = xh[i] - dd * m->d[i];
    norm3(m->e1); cross(m->d, m->e1, m->e2);
    /* h range of the used slices and the 98th percentile radius (subsampled) */
    m->hlo = (l2p(m, zl) - m->a0[2]) * m->d[2]; m->hhi = (l2p(m, zh) - m->a0[2]) * m->d[2];
    int64_t cnt = 0, hist[4096] = {0}; const double rs = 20;   /* 20 um bins */
    for (int64_t z = zl; z <= zh; z += 4) for (int64_t y = 0; y < ny; y += 2) for (int64_t x = 0; x < nx; x += 2) if (m->v[((size_t)z * ny + y) * nx + x]) {
        const double p[3] = {l2p(m, x) - m->a0[0], l2p(m, y) - m->a0[1], l2p(m, z) - m->a0[2]};
        const double h = p[0] * m->d[0] + p[1] * m->d[1] + p[2] * m->d[2];
        const double r = sqrt(fmax(0, p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - h * h));
        int b = (int)(r / rs); if (b > 4095) b = 4095;
        hist[b]++; cnt++;
    }
    int64_t acc = 0; int b = 0; for (; b < 4096 && acc < 0.98 * cnt; b++) acc += hist[b];
    m->rmax = b * rs;
    fprintf(stderr, "  axis point (%.0f, %.0f, %.0f) um, direction (%.4f, %.4f, %.4f), h %.0f .. %.0f um, r98 %.0f um\n", m->a0[0], m->a0[1], m->a0[2], m->d[0], m->d[1], m->d[2], m->hlo, m->hhi, m->rmax);
    free(A); free(cx); free(cy);
}
static float mvol_at(const mvol *m, const double P[3]) {   /* trilinear at physical P; < 0 if any corner is outside the mask */
    const double x = p2l(m, P[0]), y = p2l(m, P[1]), z = p2l(m, P[2]);
    const int64_t ix = (int64_t)floor(x), iy = (int64_t)floor(y), iz = (int64_t)floor(z);
    if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= m->n[2] || iy + 1 >= m->n[1] || iz + 1 >= m->n[0]) return -1;
    const double ux = x - ix, uy = y - iy, uz = z - iz; double s = 0;
    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        const uint8_t v = m->v[((size_t)(iz + dz) * m->n[1] + iy + dy) * m->n[2] + ix + dx]; if (!v) return -1;
        s += (dz ? uz : 1 - uz) * (dy ? uy : 1 - uy) * (dx ? ux : 1 - ux) * v;
    }
    return (float)s;
}

typedef struct {
    const mvol *c, *f; int nphi, nh, nr, Hc, Hf, sh, sp; double W, rmin, dr;
    double complex *acc[6]; pthread_mutex_t mu; atomic_int next;
} ijob;
/* ring k: coarse (C, C^2, mC) and fine (F, F^2, mF) cylinder images [nphi][nh], their FFTs, the six products accumulated */
static void *ring_worker(void *arg) {
    ijob *j = arg; const size_t N = (size_t)j->nphi * j->nh;
    double complex *b[6]; for (int k = 0; k < 6; k++) b[k] = malloc(N * sizeof **b);
    double complex *col = malloc(j->nphi * sizeof *col);
    for (;;) {
        const int k = atomic_fetch_add(&j->next, 1); if (k >= j->nr) break;
        const double r = j->rmin + k * j->dr;
        for (int t = 0; t < 6; t++) memset(b[t], 0, N * sizeof **b);
        int nf = 0, nc = 0;
        for (int i = 0; i < j->nphi; i++) {
            const double ph = 2 * M_PI * i / j->nphi;
            for (int h = 0; h < j->Hc; h++) {   /* coarse at (r, ph, hlo + h W) */
                const double hh = j->c->hlo + h * j->W, cs = cos(ph), sn = sin(ph); double P[3];
                for (int a = 0; a < 3; a++) P[a] = j->c->a0[a] + hh * j->c->d[a] + r * (cs * j->c->e1[a] + sn * j->c->e2[a]);
                const float v = mvol_at(j->c, P);
                if (v >= 0) { b[0][(size_t)i * j->nh + h] = v; b[1][(size_t)i * j->nh + h] = (double)v * v; b[2][(size_t)i * j->nh + h] = 1; nc++; }
            }
            const double phf = j->sp * ph, cs = cos(phf), sn = sin(phf);
            for (int h = 0; h < j->Hf; h++) {   /* fine at (r, sp ph, hb + sh h W) */
                const double hh = (j->sh > 0 ? j->f->hlo : j->f->hhi) + j->sh * h * j->W; double P[3];
                for (int a = 0; a < 3; a++) P[a] = j->f->a0[a] + hh * j->f->d[a] + r * (cs * j->f->e1[a] + sn * j->f->e2[a]);
                const float v = mvol_at(j->f, P);
                if (v >= 0) { b[3][(size_t)i * j->nh + h] = v; b[4][(size_t)i * j->nh + h] = (double)v * v; b[5][(size_t)i * j->nh + h] = 1; nf++; }
            }
        }
        if (nf < 50 || nc < 50) continue;
        for (int t = 0; t < 6; t++) fft2(b[t], j->nphi, j->nh, 0, col);
        /* corr(X, Y)(u) = sum_x X(x) Y(x + u) = IFFT(conj(FX) FY): F*C, F*mC, F2*mC, mF*C, mF*C2, mF*mC */
        pthread_mutex_lock(&j->mu);
        for (size_t i = 0; i < N; i++) {
            const double complex F = conj(b[3][i]), F2 = conj(b[4][i]), mF = conj(b[5][i]);
            j->acc[0][i] += F * b[0][i]; j->acc[1][i] += F * b[2][i]; j->acc[2][i] += F2 * b[2][i];
            j->acc[3][i] += mF * b[0][i]; j->acc[4][i] += mF * b[1][i]; j->acc[5][i] += mF * b[2][i];
        }
        pthread_mutex_unlock(&j->mu);
    }
    for (int k = 0; k < 6; k++) free(b[k]);
    free(col);
    return nullptr;
}

typedef struct { double ncc, uphi, uh; int sh, sp; } ipeak;
int reg_init_affine(rvol *cv, rvol *fv, double W, double rmin, int nt, int all, raff *Aout, reg_init_info *info) {
    mvol mc = {0}, mf = {0};
    if (mvol_load(cv, W, nt, &mc) || mvol_load(fv, W, nt, &mf)) return -1;
    fprintf(stderr, "coarse:"); mvol_axis(&mc); fprintf(stderr, "fine:"); mvol_axis(&mf);
    const double rmax = fmin(mc.rmax, mf.rmax);
    ijob j = {.c = &mc, .f = &mf, .W = W, .rmin = rmin, .dr = W};
    j.nr = (int)((rmax - rmin) / W); if (j.nr < 4) { fprintf(stderr, "reg init: radius range too small\n"); free(mc.v); free(mf.v); return -1; }
    j.nphi = 256; while (j.nphi < 2 * M_PI * rmax / W && j.nphi < 2048) j.nphi *= 2;
    j.Hc = (int)((mc.hhi - mc.hlo) / W) + 1; j.Hf = (int)((mf.hhi - mf.hlo) / W) + 1;
    j.nh = 64; while (j.nh < j.Hc + j.Hf) j.nh *= 2;
    const size_t N = (size_t)j.nphi * j.nh;
    fprintf(stderr, "reg init: %d rings %.0f..%.0f um, %d phi bins, h %d (coarse) / %d (fine) of %.0f um, grid %d x %d\n", j.nr, rmin, rmax, j.nphi, j.Hc, j.Hf, W, j.nphi, j.nh);
    for (int k = 0; k < 6; k++) j.acc[k] = malloc(N * sizeof **j.acc);
    double complex *col = malloc(j.nphi * sizeof *col);
    pthread_mutex_init(&j.mu, nullptr);
    ipeak best = {-2, 0, 0, 1, 1}; ipeak list[4]; int nl = 0;
    for (int o = 0; o < (all ? 4 : 1); o++) {
        j.sh = o & 1 ? -1 : 1; j.sp = o & 2 ? -1 : 1; atomic_store(&j.next, 0);
        for (int k = 0; k < 6; k++) memset(j.acc[k], 0, N * sizeof **j.acc);
        pthread_t th[32]; const int T = nt < 1 ? 1 : nt > 32 ? 32 : nt;
        for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, ring_worker, &j);
        for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
        for (int k = 0; k < 6; k++) fft2(j.acc[k], j.nphi, j.nh, 1, col);
        /* masked NCC; overlap >= 30% of the largest */
        double nmax = 0; for (size_t i = 0; i < N; i++) nmax = fmax(nmax, creal(j.acc[5][i]) / N);
        double *ncc = malloc(N * sizeof *ncc); size_t pk = 0; ncc[0] = -2;
        for (size_t i = 0; i < N; i++) {
            const double n = creal(j.acc[5][i]) / N, sfc = creal(j.acc[0][i]) / N, sf = creal(j.acc[1][i]) / N, sf2 = creal(j.acc[2][i]) / N, sc = creal(j.acc[3][i]) / N, sc2 = creal(j.acc[4][i]) / N;
            double v = -2;
            if (n >= 0.3 * nmax) { const double a = sf2 - sf * sf / n, b = sc2 - sc * sc / n; if (a > 0 && b > 0) v = (sfc - sf * sc / n) / sqrt(a * b); }
            ncc[i] = v; if (v > ncc[pk]) pk = i;
        }
        const int pi = (int)(pk / j.nh), ph = (int)(pk % j.nh);
        /* parabolic sub-bin */
        const double c0 = ncc[pk], cm = ncc[(size_t)((pi + j.nphi - 1) % j.nphi) * j.nh + ph], cp = ncc[(size_t)((pi + 1) % j.nphi) * j.nh + ph];
        const double hm = ncc[(size_t)pi * j.nh + (ph + j.nh - 1) % j.nh], hp = ncc[(size_t)pi * j.nh + (ph + 1) % j.nh];
        const double dphi = (cm > -2 && cp > -2 && cm + cp - 2 * c0 < 0) ? 0.5 * (cm - cp) / (cm + cp - 2 * c0) : 0, dh = (hm > -2 && hp > -2 && hm + hp - 2 * c0 < 0) ? 0.5 * (hm - hp) / (hm + hp - 2 * c0) : 0;
        int uh = ph; if (uh > j.nh / 2) uh -= j.nh;
        ipeak p = {c0, pi + dphi, uh + dh, j.sh, j.sp};
        /* the runner-up away from the peak (phi or h distance > 8 bins) */
        double sec = -2;
        for (size_t i = 0; i < N; i++) { const int a = (int)(i / j.nh), b = (int)(i % j.nh); int da = abs(a - pi); da = da > j.nphi / 2 ? j.nphi - da : da; int db = abs(b - ph); db = db > j.nh / 2 ? j.nh - db : db; if ((da > 8 || db > 8) && ncc[i] > sec) sec = ncc[i]; }
        fprintf(stderr, "  orientation h %+d phi %+d: NCC %.4f at phi %.2f deg, h offset %.0f um (runner-up %.4f)\n", j.sh, j.sp, c0, p.uphi * 360.0 / j.nphi, p.uh * W, sec);
        list[nl++] = p;
        if (c0 > best.ncc) best = p;
        free(ncc);
    }
    /* affine: fine local (u v h) = Bf^T (P_f - a0f); coarse local = (Rot(theta) diag(1, sp) (u, v), hc) with hc = hlo_c + uh W + sh (h - hb) */
    const double th = 2 * M_PI * best.uphi / j.nphi, hb = best.sh > 0 ? mf.hlo : mf.hhi, t3 = mc.hlo + best.uh * W - best.sh * hb;
    double L[3][3] = {{cos(th), -sin(th) * best.sp, 0}, {sin(th), cos(th) * best.sp, 0}, {0, 0, (double)best.sh}};
    double Bc[3][3], Bf[3][3];
    for (int a = 0; a < 3; a++) { Bc[a][0] = mc.e1[a]; Bc[a][1] = mc.e2[a]; Bc[a][2] = mc.d[a]; Bf[a][0] = mf.e1[a]; Bf[a][1] = mf.e2[a]; Bf[a][2] = mf.d[a]; }
    /* P_c = a0c + Bc (L Bf^T (P_f - a0f) + (0, 0, t3)) ; P = um x */
    double G[3][3] = {{0}};   /* Bc L Bf^T */
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) G[a][b] += Bc[a][k] * L[k][l] * Bf[b][l];
    raff A;
    for (int a = 0; a < 3; a++) {
        double t = mc.a0[a] + Bc[a][2] * t3;
        for (int b = 0; b < 3; b++) { A.m[a][b] = G[a][b] * fv->um / cv->um; t -= G[a][b] * mf.a0[b]; }
        A.m[a][3] = t / cv->um;
    }
    *Aout = A;
    if (info) { info->ncc = best.ncc; info->rot_deg = th * 180 / M_PI; info->hflip = best.sh < 0; info->pflip = best.sp < 0; info->n = nl; for (int k = 0; k < nl; k++) info->orient_ncc[k] = list[k].ncc; }
    fprintf(stderr, "reg init: best orientation h %+d phi %+d, NCC %.4f, rotation %.2f deg; det %.6g (expected %+.6g)\n", best.sh, best.sp, best.ncc, th * 180 / M_PI, raff_det(&A), pow(fv->um / cv->um, 3));
    for (int k = 0; k < 6; k++) free(j.acc[k]);
    free(col); free(mc.v); free(mf.v);
    return 0;
}

int cmd_reg_init(int argc, char **argv) {
    const char *out = opt(argc, argv, "-o", nullptr), *cr = opt(argc, argv, "--coarse-root", nullptr), *ck = opt(argc, argv, "--coarse", nullptr), *fr = opt(argc, argv, "--fine-root", cr), *fk = opt(argc, argv, "--fine", nullptr);
    if (!out || !cr || !ck || !fk) { fprintf(stderr, "usage: ufsm reg init --coarse-root R --coarse KEY [--fine-root R] --fine KEY [--um 150] [--rmin 1500 (um)] [--orient all|keep] [--threads 6] [--name N] -o pair.json\n"); return 2; }
    const double W = atof(opt(argc, argv, "--um", "150")), rmin = atof(opt(argc, argv, "--rmin", "1500"));
    const int nt = atoi(opt(argc, argv, "--threads", "6")), all = strcmp(opt(argc, argv, "--orient", "all"), "keep") != 0;
    rvol *cv = rvol_open(cr, ck, nullptr, atof(opt(argc, argv, "--coarse-um", "0"))), *fv = rvol_open(fr, fk, nullptr, atof(opt(argc, argv, "--fine-um", "0")));
    if (!cv || !fv) return 1;
    raff A; reg_init_info in = {0};
    if (reg_init_affine(cv, fv, W, rmin, nt, all, &A, &in)) return 1;
    FILE *o = fopen(out, "w"); if (!o) { perror(out); return 1; }
    fprintf(o, "{\"name\":\"%s\",\n \"coarse\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"fine\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"A\":[[%.12g,%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g,%.12g]],\n",
            opt(argc, argv, "--name", "pair"), cr, ck, cv->um, fr, fk, fv->um, A.m[0][0], A.m[0][1], A.m[0][2], A.m[0][3], A.m[1][0], A.m[1][1], A.m[1][2], A.m[1][3], A.m[2][0], A.m[2][1], A.m[2][2], A.m[2][3]);
    fprintf(o, " \"init\":{\"um\":%.4g,\"ncc\":%.4f,\"rotation_deg\":%.3f,\"h_flip\":%d,\"phi_flip\":%d,\"orientations_ncc\":[", W, in.ncc, in.rot_deg, in.hflip, in.pflip);
    for (int k = 0; k < in.n; k++) fprintf(o, "%s%.4f", k ? "," : "", in.orient_ncc[k]);
    fprintf(o, "]},\n \"A_note\":\"fine level-0 voxel (x,y,z) -> coarse level-0 voxel (x,y,z); from-scratch init (reg init)\"}\n");
    fclose(o); rvol_close(cv); rvol_close(fv);
    return 0;
}
