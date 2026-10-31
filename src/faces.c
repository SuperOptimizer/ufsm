/* ufsm faces core: recto / verso faces of the sheet under each wrap (src/faces.h; the command: src/faces_cli.c). */
#include "faces_core.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- one profile -> bands (faces_core.h: the same code runs on the GPU) ---- */
int faces_bands(const float *f0, int m, double t0, double h, double min_prom, double rel, fc_band *out, int kmax) {
    if (m < 5 || m > 8192 || kmax < 1) return 0;
    float f[8192]; int ex[8192];
    memcpy(f, f0, m * sizeof(float));
    return fc_bands_core(f, ex, m, t0, h, min_prom, rel, out, kmax);
}

/* ---- normals / orientation ---- */
int faces_normal(const float *X, const float *Y, const float *Z, int w, int h, size_t i, double N[3]) {
    const int r = (int)(i / w), c = (int)(i % w);
    if (!(Z[i] >= 0)) return 0;
    int cl = c, cr = c, ru = r, rd = r;
    for (int d = 2; d >= 1; d--) { if (cl == c && c - d >= 0 && Z[i - d] >= 0) cl = c - d; if (cr == c && c + d < w && Z[i + d] >= 0) cr = c + d; }
    for (int d = 2; d >= 1; d--) { if (ru == r && r - d >= 0 && Z[i - (size_t)d * w] >= 0) ru = r - d; if (rd == r && r + d < h && Z[i + (size_t)d * w] >= 0) rd = r + d; }
    if (cl == cr || ru == rd) return 0;
    const size_t a = (size_t)r * w + cl, b = (size_t)r * w + cr, u = (size_t)ru * w + c, d = (size_t)rd * w + c;
    const double du[3] = {Z[b] - Z[a], Y[b] - Y[a], X[b] - X[a]}, dv[3] = {Z[d] - Z[u], Y[d] - Y[u], X[d] - X[u]};
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (!(l > 0)) return 0;
    for (int k = 0; k < 3; k++) N[k] /= l;
    return 1;
}
int faces_orient(const float *X, const float *Y, const float *Z, int w, int h, const double (*axis_yx)[2], int n_axis) {
    const size_t n = (size_t)w * h; enum { BZ = 64 };
    double zmax = 0; for (size_t i = 0; i < n; i++) if (Z[i] > zmax) zmax = Z[i];
    const int nb = (int)(zmax / BZ) + 1; double *cy = calloc(nb, sizeof(double)), *cx = calloc(nb, sizeof(double)), *cn = calloc(nb, sizeof(double));
    if (!axis_yx) for (size_t i = 0; i < n; i++) if (Z[i] >= 0) { const int b = (int)(Z[i] / BZ); cy[b] += Y[i]; cx[b] += X[i]; cn[b]++; }
    long vote = 0;
    for (size_t i = 0; i < n; i += 7) {
        double N[3]; if (!faces_normal(X, Y, Z, w, h, i, N)) continue;
        double ay, ax;
        if (axis_yx) { int zi = (int)lround(Z[i]); if (zi >= n_axis) zi = n_axis - 1; if (zi < 0) zi = 0; ay = axis_yx[zi][0]; ax = axis_yx[zi][1]; }
        else { const int b = (int)(Z[i] / BZ); if (cn[b] < 1) continue; ay = cy[b] / cn[b]; ax = cx[b] / cn[b]; }
        const double rr = N[1] * (Y[i] - ay) + N[2] * (X[i] - ax);
        vote += rr > 0 ? 1 : rr < 0 ? -1 : 0;
    }
    free(cy); free(cx); free(cn);
    return vote >= 0 ? 1 : -1;
}

/* ---- per-vertex profiles -> candidate bands ---- */
enum { FK = 4, SUB = 4 };   /* candidate bands per vertex; profile kept at every SUB-th row / column (statistics) */
typedef struct {
    const float *X, *Y, *Z; const float *N; int w, h, r0, r1; double f, hs, R; int m; const faces_opt *o; const faces_vol *V;
    fc_band *cand; uint8_t *nc; uint8_t *prof; float *ct0; _Atomic int next; int mark;
} fc_job;
static inline void fc_ctpos(const fc_job *J, size_t i, double c[3]) { c[0] = (J->Z[i] + 0.5) * J->f - 0.5; c[1] = (J->Y[i] + 0.5) * J->f - 0.5; c[2] = (J->X[i] + 0.5) * J->f - 0.5; }
static void *fc_rows(void *arg) {
    fc_job *J = arg; float f[8192]; int ex[8192];
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = 0; c < J->w; c++) {
        const size_t i = (size_t)r * J->w + c; const float *N = &J->N[3 * i];
        if (!(J->Z[i] >= 0) || isnan(N[0])) continue;
        double p[3]; fc_ctpos(J, i, p);
        if (J->mark) {
            double lo[3], hi[3];
            for (int d = 0; d < 3; d++) { const double a = p[d] - J->R * N[d], b = p[d] + J->R * N[d]; lo[d] = fmin(a, b) - 1; hi[d] = fmax(a, b) + 1; }
            J->V->mark(J->V->ctx, lo, hi); continue;
        }
        for (int k = 0; k < J->m; k++) { const double t = -J->R + k * J->hs; f[k] = (float)J->V->sample(J->V->ctx, p[0] + t * N[0], p[1] + t * N[1], p[2] + t * N[2]); }
        J->ct0[i] = f[(J->m - 1) / 2];
        if (J->prof && r % SUB == 0 && c % SUB == 0) {
            uint8_t *q = &J->prof[((size_t)(r / SUB) * ((J->w + SUB - 1) / SUB) + c / SUB) * J->m];
            for (int k = 0; k < J->m; k++) q[k] = f[k] == f[k] ? (uint8_t)lrintf(fminf(fmaxf(f[k], 0), 255)) : 0;
        }
        J->nc[i] = (uint8_t)fc_bands_core(f, ex, J->m, -J->R, J->hs, J->o->min_prom, J->o->rel, &J->cand[i * FK], FK);
    }
    return nullptr;
}
static void fc_par(void *(*fn)(void *), void *J, int nt) {
    pthread_t th[64]; const int k = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, fn, J);
    for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
}

/* ---- robust windowed median of a float field (NaN = missing; quickselect faces_core.h fc_select) ---- */
typedef struct { const float *in; float *out; int w, h, R, minn; _Atomic int next; } fm_job;
static void *fm_rows(void *arg) {
    fm_job *J = arg; const int R = J->R; float *buf = malloc(sizeof(float) * (2 * R + 1) * (2 * R + 1));
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->h;) for (int c = 0; c < J->w; c++) {
        int k = 0;
        for (int a = r - R; a <= r + R; a++) { if (a < 0 || a >= J->h) continue; const float *row = J->in + (size_t)a * J->w;
            for (int b = c - R; b <= c + R; b++) if (b >= 0 && b < J->w && !isnan(row[b])) buf[k++] = row[b]; }
        J->out[(size_t)r * J->w + c] = k >= J->minn ? fc_select(buf, k, k / 2) : NAN;
    }
    free(buf); return nullptr;
}
/* the GPU (src/nn_faces.cu); weak: absent in CPU-only binaries */
fgpu *fgpu_open(const int64_t shape[3]) __attribute__((weak));
int fgpu_broken(const fgpu *G) __attribute__((weak));
void fgpu_set_broken(fgpu *G) __attribute__((weak));
int fgpu_grid(fgpu *G, const float *X, const float *Y, const float *Z, const float *N, int w, int h) __attribute__((weak));
int fgpu_band(fgpu *G, int r0, int r1, double f, double hs, double R, int m, double min_prom, double rel, fc_band *cand, uint8_t *nc, float *ct0, uint8_t *prof) __attribute__((weak));
int fgpu_median(fgpu *G, const float *in, float *out, int w, int h, int R, int minn) __attribute__((weak));
int fgpu_mark(fgpu *G, int r0, int r1, double f, double hs, double R, int m, uint32_t *mask) __attribute__((weak));
size_t fgpu_nchunks(const fgpu *G) __attribute__((weak));
static int fc_gpu_ok(const faces_vol *V) { return V->gpu && fgpu_broken && !fgpu_broken(V->gpu); }
static void fc_median(const faces_vol *V, const float *in, float *out, int w, int h, int R, int minn, int nt) {
    if (fc_gpu_ok(V) && fgpu_median && !fgpu_median(V->gpu, in, out, w, h, R, minn < 1 ? 1 : minn)) return;
    fm_job J = {in, out, w, h, R, minn < 1 ? 1 : minn, 0}; fc_par(fm_rows, &J, nt);
}
/* UFSM_FACES_PROF=1: per-stage wall times on stderr */
static double fc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* push-pull fill of the NaN entries of a (w x h) field over the valid mask (Z >= 0); levels halve until one value per field */
static void fc_fill(float *a, const float *Z, int w, int h) {
    if (w < 1 || h < 1) return;
    const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
    if (w == 1 && h == 1) return;
    float *b = malloc(sizeof(float) * w2 * h2); int any = 0, miss = 0;
    for (int r = 0; r < h2; r++) for (int c = 0; c < w2; c++) {
        double s = 0; int n = 0;
        for (int dr = 0; dr < 2; dr++) for (int dc = 0; dc < 2; dc++) { const int rr = 2 * r + dr, cc = 2 * c + dc; if (rr < h && cc < w && !isnan(a[(size_t)rr * w + cc])) { s += a[(size_t)rr * w + cc]; n++; } }
        b[(size_t)r * w2 + c] = n ? (float)(s / n) : NAN; any |= n > 0; miss |= n == 0;
    }
    if (any && miss && (w2 < w || h2 < h)) fc_fill(b, nullptr, w2, h2);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) { const size_t i = (size_t)r * w + c; if (isnan(a[i]) && (!Z || Z[i] >= 0)) a[i] = b[(size_t)(r / 2) * w2 + c / 2]; }
    free(b);
}

typedef struct { const fc_band *cand; const uint8_t *nc; const float *mn, *mf; int *pick; size_t n; double tol; _Atomic size_t next; } rp_job;
static void *rp_rows(void *arg) {
    rp_job *J = arg;
    for (size_t i0; (i0 = atomic_fetch_add(&J->next, 65536)) < J->n;) for (size_t i = i0; i < i0 + 65536 && i < J->n; i++) {
        if (!J->nc[i] || isnan(J->mn[i]) || isnan(J->mf[i])) continue;   /* no neighbourhood: keep the first pick */
        int best = -1; double bs = 1e300;
        for (int k = 0; k < J->nc[i]; k++) {
            const fc_band *b = &J->cand[i * FK + k]; if (isnan(b->near)) continue;
            const double dn = fabs(b->near - J->mn[i]), df = fabs(b->far - J->mf[i]);
            if (dn > J->tol || df > J->tol) continue;
            if (dn + df < bs) { bs = dn + df; best = k; }
        }
        J->pick[i] = best;
    }
    return nullptr;
}

int faces_wrap(const float *X, const float *Y, const float *Z, int w, int h, double f, double ct_um, int sign, const faces_opt *o,
               const faces_vol *V, float *RX, float *RY, float *RZ, float *VX, float *VY, float *VZ, uint8_t *conf, float *thick, faces_stats *st) {
    const size_t n = (size_t)w * h; const double hs = 0.25; const int m = (int)(2 * o->range / hs) + 1; const double R = (m - 1) * hs / 2;
    if (m > 8192) return -1;
    { const double thr = st->ct_thr; memset(st, 0, sizeof *st); st->ct_thr = thr; }
    float *N = malloc(n * 3 * sizeof(float)), *ct0 = malloc(n * sizeof(float)); fc_band *cand = malloc(n * FK * sizeof(fc_band)); uint8_t *nc = calloc(n, 1);
    const int ws = (w + SUB - 1) / SUB, hsb = (h + SUB - 1) / SUB; uint8_t *prof = calloc((size_t)ws * hsb * m, 1);
    if (!N || !ct0 || !cand || !nc || !prof) { free(N); free(ct0); free(cand); free(nc); free(prof); return -1; }
    const int prof_on = getenv("UFSM_FACES_PROF") != nullptr; double tp[8] = {0}, tq = fc_now();   /* normals mark fill sample pick medians final */
    for (size_t i = 0; i < n; i++) {
        double q[3]; ct0[i] = NAN;
        if (faces_normal(X, Y, Z, w, h, i, q)) for (int k = 0; k < 3; k++) N[3 * i + k] = (float)(sign * q[k]);
        else N[3 * i] = N[3 * i + 1] = N[3 * i + 2] = NAN;
        if (Z[i] >= 0) st->nv++;
    }
    /* the GPU path needs the chunk mirror (mark / fill) and 128^3 chunks; any device failure: the CPU from that band on. With the GPU the
       decode list is the exact set of substreams the samples read (fgpu_mark), not the profiles' bounding boxes */
    int gpu = V->mark && V->fill && fc_gpu_ok(V) && fgpu_grid && fgpu_band && !fgpu_grid(V->gpu, X, Y, Z, N, w, h);
    uint32_t *gmask = gpu && V->mark_mask && fgpu_mark && fgpu_nchunks ? malloc(fgpu_nchunks(V->gpu) * sizeof(uint32_t)) : nullptr;
    tp[0] += fc_now() - tq;
    for (int r0 = 0; r0 < h; r0 += 32) {   /* bands of rows: mark the chunks, decode, sample */
        fc_job J = {X, Y, Z, N, w, h, r0, r0 + 32 < h ? r0 + 32 : h, f, hs, R, m, o, V, cand, nc, prof, ct0, 0, 1};
        tq = fc_now();
        if (V->mark) {
            if (gmask && gpu && fc_gpu_ok(V) && !fgpu_mark(V->gpu, J.r0, J.r1, f, hs, R, m, gmask)) V->mark_mask(V->ctx, gmask);
            else fc_par(fc_rows, &J, o->nthreads);
            tp[1] += fc_now() - tq; tq = fc_now();
            if (V->fill && V->fill(V->ctx)) { free(N); free(ct0); free(cand); free(nc); free(prof); free(gmask); return -1; }
            tp[2] += fc_now() - tq; tq = fc_now();
        }
        if (gpu && fc_gpu_ok(V) && !fgpu_band(V->gpu, J.r0, J.r1, f, hs, R, m, o->min_prom, o->rel, cand, nc, ct0, prof)) { tp[3] += fc_now() - tq; continue; }
        /* the GPU failed (or is off): the CPU samples the host chunks; a fill first restores substreams only the device had (gpudec) */
        if (gpu && fgpu_set_broken) fgpu_set_broken(V->gpu);
        if (gpu && V->fill && V->fill(V->ctx)) { free(N); free(ct0); free(cand); free(nc); free(prof); free(gmask); return -1; }
        gpu = 0; J.mark = 0; atomic_store(&J.next, 0); fc_par(fc_rows, &J, o->nthreads);
        tp[3] += fc_now() - tq;
    }
    tq = fc_now();
    /* first pick: the band nearest t = 0, ties to the nearer inner face */
    int *pick = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) {
        pick[i] = -1; if (!nc[i]) continue; st->nband++;
        double bs = 1e300, ba = 1e300; int any = -1;
        for (int k = 0; k < nc[i]; k++) {   /* the nearest band whatever its thickness (statistics) */
            const fc_band *b = &cand[i * FK + k]; const double d = b->near > 0 ? b->near : b->far < 0 ? -b->far : 0;
            if (d + 0.25 * fabs(b->near) < ba) { ba = d + 0.25 * fabs(b->near); any = k; }
        }
        if (any >= 0) { const int tb = (int)((cand[i * FK + any].far - cand[i * FK + any].near) * ct_um / 5); st->thick_all[tb < 0 ? 0 : tb > 63 ? 63 : tb]++; }
        for (int k = 0; k < nc[i]; k++) {
            fc_band *b = &cand[i * FK + k]; const double th = b->far - b->near;
            if (th < o->tmin || th > o->tmax) { b->near = NAN; continue; }
            const double d = b->near > 0 ? b->near : b->far < 0 ? -b->far : 0, s = d + 0.25 * fabs(b->near);
            if (s < bs) { bs = s; pick[i] = k; }
        }
        st->npick0 += pick[i] >= 0;
    }
    tp[4] += fc_now() - tq; tq = fc_now();
    float *pn = malloc(n * sizeof(float)), *pf = malloc(n * sizeof(float)), *mn = malloc(n * sizeof(float)), *mf = malloc(n * sizeof(float));
    const int S = o->smooth < 1 ? 1 : o->smooth, minn = (2 * S + 1) * (2 * S + 1) / 5 + 1;
    for (int it = 0; it < 3; it++) {   /* consistency: medians of the picks, re-pick the band nearest them */
        for (size_t i = 0; i < n; i++) { const int k = pick[i]; pn[i] = k >= 0 ? cand[i * FK + k].near : NAN; pf[i] = k >= 0 ? cand[i * FK + k].far : NAN; }
        if (it == 2) break;
        fc_median(V, pn, mn, w, h, S, minn, o->nthreads); fc_median(V, pf, mf, w, h, S, minn, o->nthreads);
        rp_job P = {cand, nc, mn, mf, pick, n, o->tol, 0}; fc_par(rp_rows, &P, o->nthreads);
    }
    for (size_t i = 0; i < n; i++) if (pick[i] >= 0) st->npick++;
    for (size_t i = 0; i < n; i++) {   /* face texture anisotropy between picked neighbours */
        if (pick[i] < 0) continue;
        const fc_band *a = &cand[i * FK + pick[i]]; const int c = (int)(i % w);
        if (c + 1 < w && pick[i + 1] >= 0) { const fc_band *b = &cand[(i + 1) * FK + pick[i + 1]]; st->fib[0][1] += fabsf(a->cin - b->cin); st->fib[1][1] += fabsf(a->cout - b->cout); }
        if (i + w < n && pick[i + w] >= 0) { const fc_band *b = &cand[(i + w) * FK + pick[i + w]]; st->fib[0][0] += fabsf(a->cin - b->cin); st->fib[1][0] += fabsf(a->cout - b->cout); }
    }
    fc_median(V, pn, mn, w, h, 2, 3, o->nthreads); fc_median(V, pf, mf, w, h, 2, 3, o->nthreads);   /* final 5 x 5 medians; isolated misses filled */
    tp[5] += fc_now() - tq; tq = fc_now();
    for (size_t i = 0; i < n; i++) if (Z[i] < 0) mn[i] = mf[i] = NAN; else if (isnan(mn[i]) || isnan(mf[i])) { mn[i] = mf[i] = NAN; st->nfill++; }
    fc_fill(mn, Z, w, h); fc_fill(mf, Z, w, h);
    for (size_t i = 0; i < n; i++) {
        RX[i] = RY[i] = RZ[i] = VX[i] = VY[i] = VZ[i] = -1; conf[i] = 0; thick[i] = NAN;
        if (Z[i] < 0) continue;
        const float *Nv = &N[3 * i]; double nr = mn[i], nf = mf[i];
        if (isnan(Nv[0]) || isnan(nr) || isnan(nf)) continue;
        if (nf - nr < 2 * o->inset + 0.5) { const double c = 0.5 * (nr + nf); nr = nf = c; } else { nr += o->inset; nf -= o->inset; }
        const double a = nr / f, b = nf / f;
        RZ[i] = (float)(Z[i] + a * Nv[0]); RY[i] = (float)(Y[i] + a * Nv[1]); RX[i] = (float)(X[i] + a * Nv[2]);
        VZ[i] = (float)(Z[i] + b * Nv[0]); VY[i] = (float)(Y[i] + b * Nv[1]); VX[i] = (float)(X[i] + b * Nv[2]);
        thick[i] = (float)((mf[i] - mn[i]) * ct_um);
        if (pick[i] >= 0) { const double c = cand[i * FK + pick[i]].prom; conf[i] = (uint8_t)(c < 1 ? 1 : c > 255 ? 255 : c); }
        st->shift_r += fabs(a); st->shift_v += fabs(b);
        if (pick[i] >= 0) { const int tb = (int)(thick[i] / 5); st->thick_hist[tb < 0 ? 0 : tb > 63 ? 63 : tb]++; if (mn[i] <= 0 && mf[i] >= 0) st->in_band++; }
        /* the CT at the vertex and at the faces, from the kept profiles */
        const int r = (int)(i / w), c = (int)(i % w);
        if (r % SUB == 0 && c % SUB == 0 && !isnan(ct0[i])) {
            const uint8_t *q = &prof[((size_t)(r / SUB) * ws + c / SUB) * m];
            const double kr = (nr + R) / hs, kv = (nf + R) / hs;
            if (kr >= 0 && kv <= m - 1) {
                const int ir = (int)kr, iv = (int)kv; const double fr = kr - ir, fv = kv - iv;
                const double cr = q[ir] + (ir + 1 < m ? fr * (q[ir + 1] - q[ir]) : 0), cv = q[iv] + (iv + 1 < m ? fv * (q[iv + 1] - q[iv]) : 0);
                st->ct_n++; st->ct_orig += ct0[i] >= st->ct_thr; st->ct_rect += cr >= st->ct_thr; st->ct_vers += cv >= st->ct_thr;
            }
        }
    }
    tp[6] += fc_now() - tq;
    if (prof_on) fprintf(stderr, "faces prof: normals %.2f mark %.2f fill %.2f sample %.2f pick %.2f medians %.2f final %.2f s (%s)\n", tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6], gpu ? "gpu" : "cpu");
    free(N); free(ct0); free(cand); free(nc); free(prof); free(pick); free(pn); free(pf); free(mn); free(mf); free(gmask);
    return 0;
}

