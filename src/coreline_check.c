/* ufsm coreline check: the core-line sanity renders of the former core/check/check.py (see coreline.h cl_check), apart from
   coreline.c so the objects that link the core line need no image code. */
#include "coreline.h"
#include "imgops.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *chk_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---- check (the former core/check/check.py) ---- */
/* numpy's pairwise sum of n <= 128 doubles (8 accumulators, then the tail in order) */
static double np_pairwise(const double *a, int n) {
    if (n < 8) { double s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
    double r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
    int i = 8; for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) s += a[i];
    return s;
}
void scipy_gaussian_f32(float *a, int H, int W, double sigma) {
    const int r = (int)(4.0 * sigma + 0.5); double k[2 * 128 + 1], wk[129];   /* _gaussian_kernel1d: exp(-0.5 / sigma^2 x^2) / sum */
    if (r > 128) return;
    for (int i = -r; i <= r; i++) k[i + r] = exp(-0.5 / (sigma * sigma) * (double)(i * i));
    const double s = np_pairwise(k, 2 * r + 1); for (int i = 0; i <= r; i++) wk[i] = k[i + r] / s;   /* wk[j]: offset +-j (symmetric) */
    double *acc = malloc((size_t)(H > W ? H : W) * sizeof *acc); float *src = malloc((size_t)H * W * sizeof *src);
    if (!acc || !src) { free(acc); free(src); return; }
    /* NI_Correlate1D's symmetric path per output: x0 w0, then + (x[-j] + x[+j]) w_j for j = r .. 1, in double; reflect edges */
    #define REFL(i, n) ((i) < 0 ? -(i) - 1 : (i) >= (n) ? 2 * (n) - (i) - 1 : (i))
    memcpy(src, a, (size_t)H * W * sizeof *src);
    for (int y = 0; y < H; y++) {   /* axis 0 */
        const float *c0 = src + (size_t)y * W;
        for (int x = 0; x < W; x++) acc[x] = (double)c0[x] * wk[0];
        for (int j = r; j >= 1; j--) {
            const float *u = src + (size_t)REFL(y - j, H) * W, *d = src + (size_t)REFL(y + j, H) * W;
            for (int x = 0; x < W; x++) acc[x] += ((double)u[x] + (double)d[x]) * wk[j];
        }
        for (int x = 0; x < W; x++) a[(size_t)y * W + x] = (float)acc[x];
    }
    memcpy(src, a, (size_t)H * W * sizeof *src);
    for (int y = 0; y < H; y++) {   /* axis 1 */
        const float *row = src + (size_t)y * W;
        for (int x = 0; x < W; x++) {
            double v = (double)row[x] * wk[0];
            for (int j = r; j >= 1; j--) v += ((double)row[REFL(x - j, W)] + (double)row[REFL(x + j, W)]) * wk[j];
            a[(size_t)y * W + x] = (float)v;
        }
    }
    #undef REFL
    free(acc); free(src);
}
/* the |signed distance| (cells) of the crop's cell centres to the polyline, rows split over threads */
typedef struct { const double *Y, *X; int n, y0, x0, ch, cw; double *sd; atomic_int next; } chk_job;
static void *chk_sd_worker(void *p) {
    chk_job *J = p;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->ch;)
        for (int c = 0; c < J->cw; c++) J->sd[(size_t)r * J->cw + c] = fabs(cl_poly_sdist(J->Y, J->X, J->n, J->y0 + r + 0.5, J->x0 + c + 0.5, nullptr, nullptr));
    return nullptr;
}
/* numpy.gradient along one axis (edge_order 1) of a float64 h x w array: central differences inside, one-sided at the ends */
static void np_grad_f64(const double *f, int h, int w, int axis, double *g) {
    const int n = axis ? w : h; const size_t st = axis ? 1 : (size_t)w;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            const int i = axis ? c : r; const size_t k = (size_t)r * w + c;
            g[k] = n < 2 ? 0 : i == 0 ? f[k + st] - f[k] : i == n - 1 ? f[k] - f[k - st] : (f[k + st] - f[k - st]) / 2.0;
        }
}
int cl_check(const coreline *C, int zc, const uint8_t *ct, int ctw, int cth, const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int N,
             int nthreads, cl_check_res *o) {
    memset(o, 0, sizeof *o);
    const size_t nn = (size_t)N * N;
    float *gy = calloc(nn, 4), *gx = calloc(nn, 4), *wm = malloc(nn * 4), *A = malloc(nn * 4), *B = malloc(nn * 4);
    if (!gy || !gx || !wm || !A || !B) { free(gy); free(gx); free(wm); free(A); free(B); return -2; }
    for (int k = 0; k < 8; k++) {   /* g += Im(conj(m) grad m) = a db - b da per row, complex64 as numpy computes it */
        const uint8_t *P = pc + k * nn, *S = ps + k * nn;
        for (size_t i = 0; i < nn; i++) { A[i] = (float)(P[i] - 128) / 127.f; B[i] = (float)(S[i] - 128) * (1.f / 127.f); }   /* 1j * (ps - 128) / 127: a complex division, x (1 / 127) */
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                const size_t i = (size_t)y * N + x;
                const float day = y == 0 ? A[i + N] - A[i] : y == N - 1 ? A[i] - A[i - N] : (A[i + N] - A[i - N]) / 2.f;
                const float dby = y == 0 ? B[i + N] - B[i] : y == N - 1 ? B[i] - B[i - N] : (B[i + N] - B[i - N]) / 2.f;
                const float dax = x == 0 ? A[i + 1] - A[i] : x == N - 1 ? A[i] - A[i - 1] : (A[i + 1] - A[i - 1]) / 2.f;
                const float dbx = x == 0 ? B[i + 1] - B[i] : x == N - 1 ? B[i] - B[i - 1] : (B[i + 1] - B[i - 1]) / 2.f;
                gy[i] += fmaf(A[i], dby, -(B[i] * day)); gx[i] += fmaf(A[i], dbx, -(B[i] * dax));   /* numpy's complex64 multiply (FMA) */
            }
    }
    free(A); free(B);
    for (size_t i = 0; i < nn; i++) { float s = 0; for (int k = 0; k < 8; k++) s += (float)w[k * nn + i]; wm[i] = s / 8.f / 255.f; }
    scipy_gaussian_f32(gy, N, N, 3); scipy_gaussian_f32(gx, N, N, 3); scipy_gaussian_f32(wm, N, N, 2);
    int ymn = N, ymx = -1, xmn = N, xmx = -1;
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) if (wm[(size_t)y * N + x] > 0.15) { ymn = y < ymn ? y : ymn; ymx = y > ymx ? y : ymx; xmn = x < xmn ? x : xmn; xmx = x > xmx ? x : xmx; }
    if (ymx < 0) { free(gy); free(gx); free(wm); return -1; }
    int ks = 0; for (int k = 1; k < C->ns; k++) if (fabs(C->s[k].zc / 8 - (zc + 4)) < fabs(C->s[ks].zc / 8 - (zc + 4))) ks = k;
    o->ks = ks; o->y0 = ymn - 20 > 0 ? ymn - 20 : 0; o->x0 = xmn - 20 > 0 ? xmn - 20 : 0; o->y1 = ymx + 20; o->x1 = xmx + 20;
    const int y0 = o->y0, x0 = o->x0, ch = (o->y1 < N ? o->y1 : N) - y0, cw = (o->x1 < N ? o->x1 : N) - x0; const size_t nc = (size_t)ch * cw;
    const cl_slab *S = &C->s[ks]; double *Y = malloc(S->n * sizeof *Y), *X = malloc(S->n * sizeof *X);
    double *sd = malloc(nc * 8), *ay = malloc(nc * 8), *ax = malloc(nc * 8), *a = malloc(nc * 8), *v = malloc(nc * 8);
    uint8_t *away = calloc(nc * 3, 1);
    int rc = -2;
    if (!Y || !X || !sd || !ay || !ax || !a || !v || !away) goto out;
    for (int i = 0; i < S->n; i++) { Y[i] = S->y[i] / 8; X[i] = S->x[i] / 8; }
    {
        chk_job J = {Y, X, S->n, y0, x0, ch, cw, sd, 0};
        const int nt = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; pthread_t th[64];
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, chk_sd_worker, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    }
    np_grad_f64(sd, ch, cw, 0, ay); np_grad_f64(sd, ch, cw, 1, ax);
    size_t nm = 0;
    for (int r = 0; r < ch; r++)
        for (int c = 0; c < cw; c++) {
            const size_t k = (size_t)r * cw + c, g = (size_t)(y0 + r) * N + x0 + c;
            a[k] = (double)gy[g] * ay[k] + (double)gx[g] * ax[k];
            if (wm[g] > 0.15) {
                v[nm++] = fabs(a[k]);
                if (a[k] > 0) o->pos++; else if (a[k] < 0) o->neg++;
                if (sd[k] < 26) { if (a[k] > 0) o->pn++; else if (a[k] < 0) o->nn++; }
            }
        }
    const double p90 = nm ? np_percentile_f64(v, nm, 90) : 0;
    for (int r = 0; r < ch; r++)
        for (int c = 0; c < cw; c++) {
            const size_t k = (size_t)r * cw + c; if (!(wm[(size_t)(y0 + r) * N + x0 + c] > 0.15)) continue;
            double t = a[k] / (p90 + 1e-9); t = t < -1 ? -1 : t > 1 ? 1 : t;
            const double re = -t < 0 ? 0 : -t, gr = t < 0 ? 0 : t;
            away[k * 3] = (uint8_t)(re * 255); away[k * 3 + 1] = (uint8_t)(gr * 255); away[k * 3 + 2] = 30;
        }
    {   /* the CT: 2 x 2 means (float32), cropped like the cells, contrast-stretched to its 99.5th percentile above 20 */
        const int H2 = cth / 2, W2 = ctw / 2, eh = y0 < H2 ? ((o->y1 < H2 ? o->y1 : H2) - y0) : 0, ew = x0 < W2 ? ((o->x1 < W2 ? o->x1 : W2) - x0) : 0;
        const int th = eh > 0 ? eh : 0, tw = ew > 0 ? ew : 0;
        float *cf = malloc(((size_t)th * tw + 1) * 4), *pos = malloc(((size_t)th * tw + 1) * 4); size_t np = 0;
        if (!cf || !pos) { free(cf); free(pos); goto out; }
        for (int r = 0; r < th; r++)
            for (int c = 0; c < tw; c++) {
                const int yy = 2 * (y0 + r), xx = 2 * (x0 + c);
                const float s = (float)ct[(size_t)yy * ctw + xx] + (float)ct[(size_t)yy * ctw + xx + 1] + (float)ct[(size_t)(yy + 1) * ctw + xx] + (float)ct[(size_t)(yy + 1) * ctw + xx + 1];
                const float m = s / 4.f; cf[(size_t)r * tw + c] = m; if (m > 0) pos[np++] = m;
            }
        const float p = np ? np_percentile_f32(pos, np, 99.5f) : 0.f, den = p - 20.f + 1e-9f;
        free(pos);
        o->w = tw * 2 + 8; o->h = th; o->rgb = calloc((size_t)o->w * (o->h > 0 ? o->h : 1) * 3, 1);
        if (!o->rgb) { free(cf); goto out; }
        uint8_t *ctp = malloc((size_t)tw * th * 3 + 1), *awp = malloc(nc * 3 + 1);
        if (!ctp || !awp) { free(cf); free(ctp); free(awp); goto out; }
        for (size_t i = 0; i < (size_t)th * tw; i++) {
            float t = cf[i];
            if (np) { t = (cf[i] - 20.f) * 255.f / den; t = t < 0 ? 0 : t > 255 ? 255 : t; }
            ctp[3 * i] = ctp[3 * i + 1] = ctp[3 * i + 2] = (uint8_t)t;
        }
        free(cf); memcpy(awp, away, nc * 3);
        static const uint8_t cyan[3] = {0, 200, 255}, yellow[3] = {255, 230, 0};
        const int kk[3] = {ks - 1, ks + 1, ks};
        for (int q = 0; q < 3; q++) {   /* the lines on both panels (ImageDraw.line width 2: integer-truncated points) */
            if (kk[q] < 0 || kk[q] >= C->ns) continue;
            const cl_slab *L = &C->s[kk[q]]; const uint8_t *col = q == 2 ? yellow : cyan;
            for (int i = 0; i + 1 < L->n; i++) {
                const int px0 = (int)(L->x[i] / 8 - x0), py0 = (int)(L->y[i] / 8 - y0), px1 = (int)(L->x[i + 1] / 8 - x0), py1 = (int)(L->y[i + 1] / 8 - y0);
                img_line2(ctp, tw, th, px0, py0, px1, py1, col); img_line2(awp, cw, ch, px0, py0, px1, py1, col);
            }
        }
        for (int r = 0; r < th; r++) {
            memcpy(o->rgb + (size_t)r * o->w * 3, ctp + (size_t)r * tw * 3, (size_t)tw * 3);
            if (r < ch) { const int n2 = cw < o->w - tw - 8 ? cw : o->w - tw - 8; memcpy(o->rgb + ((size_t)r * o->w + tw + 8) * 3, awp + (size_t)r * cw * 3, (size_t)n2 * 3); }
        }
        free(ctp); free(awp);
        char cap[512]; snprintf(cap, sizeof cap, "z cells %d (2.4um z %d), slab %.0f-%.0f, away share %.2f, within 0.5 mm of line %.2f", zc, zc * 8, S->z0 / 8, S->z1 / 8,
                                o->pos / (o->pos + o->neg + 1e-9), o->pn / (o->pn + o->nn + 1e-9));
        static const uint8_t ycap[3] = {255, 255, 0};
        img_text(o->rgb, o->w, o->h, 10, 10, cap, ycap);
    }
    rc = 0;
out:
    free(gy); free(gx); free(wm); free(Y); free(X); free(sd); free(ay); free(ax); free(a); free(v); free(away);
    return rc;
}

/* Python's repr(round(x, 3)) */
static void py_round3(double x, char *b, size_t cap) {
    snprintf(b, cap, "%.3f", x); size_t L = strlen(b);
    while (L > 0 && b[L - 1] == '0' && L >= 2 && b[L - 2] != '.') b[--L] = 0;
}
static uint8_t *chk_read(const char *path, size_t n) {   /* exactly n bytes */
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    uint8_t *b = malloc(n ? n : 1); const size_t r = b ? fread(b, 1, n, f) : 0; fclose(f);
    if (r != n) { free(b); return nullptr; }
    return b;
}
/* ufsm coreline check <core.json> <scratch-dir> <out-dir> ZCELL... [--n 2573] [--threads 8] [--quality 85]: the former check.py.
   Per field-cell z: <scratch-dir>/ct_Z.pgm (ufsm slice, the CT at level 2) and pc_Z.raw / ps_Z.raw / w_Z.raw (ufsm read, 8 rows
   of N x N cells from z) -> <out-dir>/zZZZZ.jpg (cl_check) and one line on stdout as check.py printed it. */
int cmd_coreline_check(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: ufsm coreline check <core.json> <scratch-dir> <out-dir> ZCELL... [--n 2573] [--threads 8] [--quality 85]\n"); return 2; }
    const int N = atoi(chk_opt(argc, argv, "--n", "2573")), nt = atoi(chk_opt(argc, argv, "--threads", "8")), qual = atoi(chk_opt(argc, argv, "--quality", "85"));
    if (N < 2) { fprintf(stderr, "coreline check: --n >= 2\n"); return 2; }
    coreline C; if (cl_load(&C, argv[3])) { fprintf(stderr, "coreline: cannot load %s\n", argv[3]); return 1; }
    const char *S = argv[4], *od = argv[5]; int rc = 0;
    for (int i = 6; i < argc; i++) {
        if (!strncmp(argv[i], "--", 2)) { i++; continue; }
        const int zc = atoi(argv[i]); char p[PATH_MAX];
        snprintf(p, sizeof p, "%s/ct_%d.pgm", S, zc); FILE *f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "coreline check: cannot read %s\n", p); rc = 1; continue; }
        fseek(f, 0, SEEK_END); const long fl = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *raw = malloc(fl > 0 ? (size_t)fl : 1); const size_t got = raw ? fread(raw, 1, (size_t)fl, f) : 0; fclose(f);
        int ctw = 0, cth = 0; const uint8_t *ct = nullptr;
        {   /* the header as check.py split it: P5 \n W H \n maxval \n data (the last W x H bytes) */
            const uint8_t *a = memchr(raw, '\n', got), *b = a ? memchr(a + 1, '\n', got - (a + 1 - raw)) : nullptr, *c = b ? memchr(b + 1, '\n', got - (b + 1 - raw)) : nullptr;
            if (c && sscanf((const char *)a + 1, "%d %d", &ctw, &cth) == 2 && ctw > 0 && cth > 0 && (size_t)(got - (c + 1 - raw)) >= (size_t)ctw * cth) ct = raw + got - (size_t)ctw * cth;
        }
        const size_t n8 = (size_t)8 * N * N; uint8_t *fp[3] = {0}; static const char *nm[3] = {"pc", "ps", "w"};
        for (int k = 0; k < 3; k++) { snprintf(p, sizeof p, "%s/%s_%d.raw", S, nm[k], zc); fp[k] = chk_read(p, n8); }
        if (!ct || !fp[0] || !fp[1] || !fp[2]) { fprintf(stderr, "coreline check: z %d: bad or missing inputs in %s\n", zc, S); free(raw); for (int k = 0; k < 3; k++) free(fp[k]); rc = 1; continue; }
        cl_check_res o; const int e = cl_check(&C, zc, ct, ctw, cth, fp[0], fp[1], fp[2], N, nt, &o);
        free(raw); for (int k = 0; k < 3; k++) free(fp[k]);
        if (e) { fprintf(stderr, "coreline check: z %d: %s\n", zc, e == -1 ? "no material" : "out of memory"); rc = 1; continue; }
        snprintf(p, sizeof p, "%s/z%04d.jpg", od, zc);
        if (o.h < 1 || jpeg_write_rgb(p, o.w, o.h, o.rgb, qual)) { fprintf(stderr, "coreline check: cannot write %s\n", p); rc = 1; }
        free(o.rgb);
        char s1[32], s2[32]; py_round3(o.pos / (o.pos + o.neg + 1e-9), s1, sizeof s1); py_round3(o.pn / (o.pn + o.nn + 1e-9), s2, sizeof s2);
        printf("%d slab %d away share %s near line %s crop %d %d %d %d\n", zc, o.ks, s1, s2, o.y0, o.y1, o.x0, o.x1); fflush(stdout);
    }
    cl_free(&C);
    return rc;
}

