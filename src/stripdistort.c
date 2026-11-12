/* ufsm strip-distort <flat-dir> <out-dir> [--div 8] [--span 4] [--tile-cm 1] [--trust 0.05] [--quality 88] [--threads N] [--stripes a,b,..]
   The flattening distortion of a strip-flatten run, per stripe, so ink review knows where the flat image keeps shapes and areas.
   Input: <flat-dir>/stripes.tsv + stripes/<s>/{x,y,z,gu}.tif (the stripe's 3-D positions on its isometric flat uv grid, -1 = none;
   gu = the source strip column, which <flat-dir>/../strip.tsv maps to a wrap) and flat.json (um, ostep = native voxels per grid step).
   Per grid pixel: the 3x2 Jacobian J = [dX/du dX/dv] / ostep (1 = isometric) by central differences over +-(--span) px (4: 0.6 mm at
   38.4 um px; one px would measure the sfc position error, 6 % per edge, not the flattening), its singular
   values s1 >= s2 (sqrt of the metric tensor's eigenvalues), and
     dist = max(s1, 1 / s2) - 1   the worst linear stretch or compression in any direction (0.05: a letter 5 % off in some direction),
     area = s1 s2                 the area ratio flat -> scroll (1 = equal area),
     fold = the surface normal du x dv points against the summed normals of the 9 x 9 --div blocks around it (a flattening flip: the
            flat image is mirrored there).
   Writes per stripe <s>_dist.jpg (the block mean of dist per --div x --div block: black none, green <= --trust, yellow 2 x --trust, red
   >= 4 x --trust; magenta: >= 10 % of the block folded) and <s>_area.jpg (block mean log2 area: blue compressed, grey 1, red stretched, +-0.5 full
   scale), at 1 / (div x render --scale 1 grid px) of the stripe render (img_fill/<s>_ct.jpg has ostep x px_scale px per grid step);
   tiles.tsv (per --tile-cm square of each stripe: its wrap, valid cm2, dist p50 / p95, area p50, the trusted share dist <= --trust,
   folds), wraps.tsv (per wrap: the same over its pixels) and stripes.tsv (per stripe); prints the totals. CPU, rows on --threads. */
#include "imgops.h"
#include "sysinfo.h"
#include "tiff.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *sd_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double sd_num(const char *b, const char *k, double d) { const char *q = strstr(b, k); if (!q) return d; q = strchr(q + strlen(k), ':'); return q ? atof(q + 1) : d; }

/* one float page of a tif (w x h), nullptr on failure */
static float *sd_read(const char *dir, const char *name, int *w, int *h) {
    char p[1600]; snprintf(p, sizeof p, "%s/%s.tif", dir, name);
    tiff *t = tiff_open_file(p); if (!t) return nullptr;
    tiff_page pi; float *v = nullptr;
    if (!tiff_page_info(t, 0, &pi) && pi.bits == 32 && pi.fmt == 3 && pi.spp == 1) {
        v = malloc((size_t)pi.w * pi.h * 4); if (v && tiff_read_page(t, 0, v)) { free(v); v = nullptr; }
        *w = pi.w; *h = pi.h;
    }
    tiff_close(t); return v;
}

/* per pixel results: dist (float, NAN none), log2 area, fold flag; the wrap per pixel (int16, -1 none) */
typedef struct {
    const float *x, *y, *z, *gu; int w, h; double step;
    float *dist, *la; uint8_t *fold; int16_t *wrap;
    const int *segc0, *segwrap; int nseg;
    int div, bw, bh, span, pass; float *bn; _Atomic int next;
} sd_job;
static int sd_wrap_of(const sd_job *J, double g) {   /* the strip segment holding source column g (segments by col0 ascending) */
    if (!(g >= 0) || !J->nseg) return INT16_MIN;
    int lo = 0, hi = J->nseg - 1;
    while (lo < hi) { const int m = (lo + hi + 1) / 2; if (J->segc0[m] <= g) lo = m; else hi = m - 1; }
    return J->segwrap[lo];
}
static inline int sd_ok(const sd_job *J, size_t i) { return J->z[i] >= 0 && J->x[i] >= 0 && J->y[i] >= 0; }
static void *sd_rows(void *arg) {
    sd_job *J = arg; const int w = J->w;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->h;) {
        for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; J->dist[i] = NAN; J->la[i] = NAN; J->fold[i] = 0; J->wrap[i] = INT16_MIN;
            if (!sd_ok(J, i)) continue;
            J->wrap[i] = (int16_t)sd_wrap_of(J, J->gu ? J->gu[i] : -1);
            /* central differences over +-span px (the stripe positions carry the strip's sfc error, 0.25 voxel: over one 4-voxel
               step that is 6 % noise per edge, over +-4 steps ~2 %); one-sided at holes and edges */
            const int S = J->span; int ca = c, cb = c, ra = r, rb = r;
            for (int d = S; d >= 1; d--) if (c + d < w && sd_ok(J, i + d)) { cb = c + d; break; }
            for (int d = S; d >= 1; d--) if (c - d >= 0 && sd_ok(J, i - d)) { ca = c - d; break; }
            for (int d = S; d >= 1; d--) if (r + d < J->h && sd_ok(J, i + (size_t)d * w)) { rb = r + d; break; }
            for (int d = S; d >= 1; d--) if (r - d >= 0 && sd_ok(J, i - (size_t)d * w)) { ra = r - d; break; }
            if (cb == ca || rb == ra) continue;
            const size_t ia = (size_t)r * w + ca, ib = (size_t)r * w + cb, ja = (size_t)ra * w + c, jb = (size_t)rb * w + c; const double su = (cb - ca) * J->step, sv = (rb - ra) * J->step;
            const double du[3] = {(J->x[ib] - J->x[ia]) / su, (J->y[ib] - J->y[ia]) / su, (J->z[ib] - J->z[ia]) / su};
            const double dv[3] = {(J->x[jb] - J->x[ja]) / sv, (J->y[jb] - J->y[ja]) / sv, (J->z[jb] - J->z[ja]) / sv};
            const double a = du[0] * du[0] + du[1] * du[1] + du[2] * du[2], cc = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2], b = du[0] * dv[0] + du[1] * dv[1] + du[2] * dv[2];
            const double m = 0.5 * (a + cc), q = sqrt(fmax(0, 0.25 * (a - cc) * (a - cc) + b * b)), l1 = m + q, l2 = fmax(0, m - q);
            const double s1 = sqrt(l1), s2 = sqrt(l2);
            if (!(s2 > 1e-6)) { J->dist[i] = 10; J->la[i] = -10; continue; }
            J->dist[i] = (float)(fmax(s1, 1 / s2) - 1); J->la[i] = (float)log2(s1 * s2);
        }
    }
    return nullptr;
}
/* folds: a pixel whose normal du x dv points against the summed unit normals of the 9 x 9 --div blocks around its block (a flattening
   flip: the flat image is mirrored there; a fold narrower than ~4 blocks is outvoted by its unfolded surroundings). Pass 0: per block
   the sum of its unit normals (bn), pass 1: the flags. */
static inline int sd_normal(const sd_job *J, size_t i, int r, int c, double n[3]) {
    const int w = J->w; if (!(J->dist[i] == J->dist[i]) || c + 1 >= w || r + 1 >= J->h || !sd_ok(J, i + 1) || !sd_ok(J, i + w)) return 0;
    const double du[3] = {J->x[i + 1] - J->x[i], J->y[i + 1] - J->y[i], J->z[i + 1] - J->z[i]}, dv[3] = {J->x[i + w] - J->x[i], J->y[i + w] - J->y[i], J->z[i + w] - J->z[i]};
    n[0] = du[1] * dv[2] - du[2] * dv[1]; n[1] = du[2] * dv[0] - du[0] * dv[2]; n[2] = du[0] * dv[1] - du[1] * dv[0]; return 1;
}
static void *sd_folds(void *arg) {
    sd_job *J = arg; const int w = J->w, D = J->div;
    for (int br; (br = atomic_fetch_add(&J->next, 1)) < J->bh;) for (int bc = 0; bc < J->bw; bc++) {
        const int r0 = br * D, c0 = bc * D, r1 = r0 + D < J->h ? r0 + D : J->h, c1 = c0 + D < w ? c0 + D : w;
        if (J->pass == 0) {
            double nm[3] = {0, 0, 0};
            for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) { double n[3]; if (!sd_normal(J, (size_t)r * w + c, r, c, n)) continue;
                const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (l > 0) for (int d = 0; d < 3; d++) nm[d] += n[d] / l; }
            for (int d = 0; d < 3; d++) J->bn[((size_t)br * J->bw + bc) * 3 + d] = (float)nm[d];
        } else {
            double ref[3] = {0, 0, 0};
            for (int a = br - 4; a <= br + 4; a++) for (int b = bc - 4; b <= bc + 4; b++) if (a >= 0 && b >= 0 && a < J->bh && b < J->bw) for (int d = 0; d < 3; d++) ref[d] += J->bn[((size_t)a * J->bw + b) * 3 + d];
            for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) { double n[3]; const size_t i = (size_t)r * w + c; if (sd_normal(J, i, r, c, n)) J->fold[i] = n[0] * ref[0] + n[1] * ref[1] + n[2] * ref[2] < 0; }
        }
    }
    return nullptr;
}
static void sd_par(sd_job *J, void *(*fn)(void *), int nt) { pthread_t th[64]; atomic_store(&J->next, 0); for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, fn, J); for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr); }

/* histogram-based quantiles of dist over [0, 1) in 1/4000 steps (+ an overflow bin) */
enum { SD_NB = 4001 };
typedef struct { double n, trust, fold, las; long long hd[SD_NB], ha[SD_NB]; } sd_acc;
static void sd_add(sd_acc *A, double d, double la, int fold, double trust) {
    A->n++; A->trust += d <= trust; A->fold += fold; A->las += la;
    int b = (int)(d * 4000); A->hd[b < 0 ? 0 : b >= SD_NB ? SD_NB - 1 : b]++;
    b = (int)((la + 1) * 2000); A->ha[b < 0 ? 0 : b >= SD_NB ? SD_NB - 1 : b]++;
}
static double sd_q(const long long *h, double n, double q, double scale, double off) { double s = 0; for (int b = 0; b < SD_NB; b++) { s += h[b]; if (s >= q * n) return (b + 0.5) / scale + off; } return (SD_NB - 0.5) / scale + off; }
static void sd_row(FILE *f, const sd_acc *A, double cm2px) {
    fprintf(f, "\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.5f", A->n * cm2px, sd_q(A->hd, A->n, 0.5, 4000, 0), sd_q(A->hd, A->n, 0.95, 4000, 0),
            exp2(sd_q(A->ha, A->n, 0.5, 2000, -1)), exp2(A->las / A->n), A->trust / A->n, A->fold / A->n);
}
static const char *SD_HDR = "valid_cm2\tdist_p50\tdist_p95\tarea_p50\tarea_gmean\ttrusted\tfolded";

static void sd_ramp(double t, uint8_t *c) {   /* t = dist / (4 trust): green up to 0.25 (trusted), to yellow at 0.5 (2 x trust), red from 1 */
    t = fmin(1, fmax(0, t));
    if (t <= 0.25) { c[0] = 40; c[1] = 170; c[2] = 60; }
    else if (t < 0.5) { const double u = (t - 0.25) / 0.25; c[0] = (uint8_t)lround(40 + 215 * u); c[1] = (uint8_t)lround(170 + 50 * u); c[2] = (uint8_t)lround(60 - 20 * u); }
    else { const double u = (t - 0.5) / 0.5; c[0] = 255; c[1] = (uint8_t)lround(220 * (1 - u)); c[2] = 40; }
}

int cmd_strip_distort(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm strip-distort <flat-dir> <out-dir> [--div 8] [--span 4] [--tile-cm 1] [--trust 0.05] [--quality 88] [--threads N] [--stripes a,b,..]\n"
                                    "  per stripe of a strip-flatten run: the worst linear distortion, the area ratio and folds (<s>_dist.jpg, <s>_area.jpg, tiles / wraps / stripes .tsv)\n"); return 2; }
    const char *fd = argv[2], *od = argv[3], *only = sd_opt(argc, argv, "--stripes", nullptr);
    const int div = atoi(sd_opt(argc, argv, "--div", "8")), q = atoi(sd_opt(argc, argv, "--quality", "88")), span = atoi(sd_opt(argc, argv, "--span", "4"));
    const double tilecm = atof(sd_opt(argc, argv, "--tile-cm", "1")), trust = atof(sd_opt(argc, argv, "--trust", "0.05"));
    int nt = ufsm_threads(atoi(sd_opt(argc, argv, "--threads", "0")), 12); if (nt > 64) nt = 64;
    if (div < 1 || span < 1 || !(tilecm > 0) || !(trust > 0)) { fprintf(stderr, "strip-distort: --div >= 1, --tile-cm > 0, --trust > 0\n"); return 2; }
    char p[1600]; snprintf(p, sizeof p, "%s/flat.json", fd); FILE *f = fopen(p, "r"); char jb[8192] = {0};
    if (!f) { fprintf(stderr, "strip-distort: no %s\n", p); return 1; }
    { const size_t nr = fread(jb, 1, sizeof jb - 1, f); (void)nr; fclose(f); }
    const double um = sd_num(jb, "\"um\"", 0), ostep = sd_num(jb, "\"ostep\"", 0);
    if (!(um > 0) || !(ostep > 0)) { fprintf(stderr, "strip-distort: no um / ostep in %s\n", p); return 1; }
    const double pxum = um * ostep, cm2px = pxum * pxum * 1e-8; const int tpx = (int)lround(tilecm * 1e4 / pxum);
    /* the strip's segments: source column -> wrap */
    int nseg = 0, *segc0 = malloc(65536 * sizeof(int)), *segw = malloc(65536 * sizeof(int)); char line[4096];
    snprintf(p, sizeof p, "%s/../strip.tsv", fd); f = fopen(p, "r");
    if (f) { if (fgets(line, sizeof line, f)) while (nseg < 65536 && fgets(line, sizeof line, f)) { int sg, wr, c0; char nm[256]; if (sscanf(line, "%d %d %255s %d", &sg, &wr, nm, &c0) == 4) { segc0[nseg] = c0; segw[nseg] = wr; nseg++; } } fclose(f); }
    else fprintf(stderr, "strip-distort: no %s: no wrap per pixel\n", p);
    snprintf(p, sizeof p, "mkdir -p '%s'", od); if (system(p)) return 1;
    snprintf(p, sizeof p, "%s/stripes.tsv", fd); f = fopen(p, "r"); if (!f) { fprintf(stderr, "strip-distort: no %s\n", p); return 1; }
    char (*names)[64] = malloc(4096 * 64); int ns = 0;
    if (fgets(line, sizeof line, f)) while (ns < 4096 && fgets(line, sizeof line, f)) if (sscanf(line, "%63s", names[ns]) == 1) ns++;
    fclose(f);
    FILE *ft, *fs, *fw; snprintf(p, sizeof p, "%s/tiles.tsv", od); ft = fopen(p, "w"); snprintf(p, sizeof p, "%s/stripes.tsv", od); fs = fopen(p, "w"); snprintf(p, sizeof p, "%s/wraps.tsv", od); fw = fopen(p, "w");
    if (!ft || !fs || !fw) { fprintf(stderr, "strip-distort: cannot write into %s\n", od); return 1; }
    fprintf(ft, "stripe\ttile_col\ttile_row\tu_cm\tv_cm\twrap(-9999 none)\t%s\n", SD_HDR); fprintf(fs, "stripe\twidth\theight\t%s\n", SD_HDR); fprintf(fw, "wrap\t%s\n", SD_HDR);
    sd_acc *T = calloc(1, sizeof *T), *WA = calloc(2048, sizeof *WA), *SA = calloc(1, sizeof *SA), *TA = nullptr;
    printf("strip-distort: span +-%d px, %d stripes, %.1f um per grid px, blocks of %d px (%.2f mm), tiles %.2f cm (%d px), trusted: dist <= %.3f\n", span, ns, pxum, div, div * pxum / 1000, tilecm, tpx, trust);
    for (int s = 0; s < ns; s++) {
        if (only) { char key[80]; snprintf(key, sizeof key, ",%s,", names[s]); char lst[4096]; snprintf(lst, sizeof lst, ",%s,", only); if (!strstr(lst, key)) continue; }
        char sdir[1600]; snprintf(sdir, sizeof sdir, "%s/stripes/%s", fd, names[s]);
        int w = 0, h = 0, w2, h2; float *X = sd_read(sdir, "x", &w, &h), *Y = sd_read(sdir, "y", &w2, &h2), *Z = sd_read(sdir, "z", &w2, &h2), *G = sd_read(sdir, "gu", &w2, &h2);
        if (!X || !Y || !Z) { fprintf(stderr, "strip-distort: cannot read %s/{x,y,z}.tif\n", sdir); return 1; }
        const size_t n = (size_t)w * h;
        sd_job J = {.x = X, .y = Y, .z = Z, .gu = G, .w = w, .h = h, .step = ostep, .dist = malloc(n * 4), .la = malloc(n * 4), .fold = malloc(n), .wrap = malloc(n * 2), .segc0 = segc0, .segwrap = segw, .nseg = nseg, .div = div, .span = span};
        J.bw = (w + div - 1) / div; J.bh = (h + div - 1) / div;
        J.bn = calloc((size_t)J.bw * J.bh * 3, sizeof(float));
        sd_par(&J, sd_rows, nt); J.pass = 0; sd_par(&J, sd_folds, nt); J.pass = 1; sd_par(&J, sd_folds, nt); free(J.bn);
        free(X); free(Y); free(Z); free(G);
        /* the block images */
        uint8_t *di = calloc((size_t)J.bw * J.bh * 3, 1), *ai = calloc((size_t)J.bw * J.bh * 3, 1);
        for (int br = 0; br < J.bh; br++) for (int bc = 0; bc < J.bw; bc++) {
            double mx = 0, las = 0; int nv = 0, nf = 0;
            for (int r = br * div; r < (br + 1) * div && r < h; r++) for (int c = bc * div; c < (bc + 1) * div && c < w; c++) {
                const size_t i = (size_t)r * w + c; if (!(J.dist[i] == J.dist[i])) continue; nv++; nf += J.fold[i]; mx += J.dist[i]; las += J.la[i]; }
            uint8_t *dp = di + 3 * ((size_t)br * J.bw + bc), *ap = ai + 3 * ((size_t)br * J.bw + bc);
            if (!nv) continue;
            if (nf * 10 >= nv) { dp[0] = 255; dp[1] = 0; dp[2] = 255; } else sd_ramp(mx / nv / (4 * trust), dp);
            const double t = fmin(1, fmax(-1, las / nv / 0.5));
            if (t >= 0) { ap[0] = (uint8_t)lround(128 + 127 * t); ap[1] = ap[2] = (uint8_t)lround(128 - 100 * t); } else { ap[2] = (uint8_t)lround(128 - 127 * t); ap[0] = ap[1] = (uint8_t)lround(128 + 100 * t); }
        }
        snprintf(p, sizeof p, "%s/%s_dist.jpg", od, names[s]); if (jpeg_write_rgb(p, J.bw, J.bh, di, q)) fprintf(stderr, "strip-distort: cannot write %s\n", p);
        snprintf(p, sizeof p, "%s/%s_area.jpg", od, names[s]); if (jpeg_write_rgb(p, J.bw, J.bh, ai, q)) fprintf(stderr, "strip-distort: cannot write %s\n", p);
        free(di); free(ai);
        /* tiles, wraps, the stripe */
        const int tw = (w + tpx - 1) / tpx, th = (h + tpx - 1) / tpx; TA = realloc(TA, (size_t)tw * th * sizeof *TA); memset(TA, 0, (size_t)tw * th * sizeof *TA);
        int *twr = calloc((size_t)tw * th * 4, sizeof(int)); memset(SA, 0, sizeof *SA);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; if (!(J.dist[i] == J.dist[i])) continue;
            const double d = J.dist[i], la = J.la[i]; const int fo = J.fold[i], ti = (r / tpx) * tw + c / tpx;
            sd_add(&TA[ti], d, la, fo, trust); sd_add(SA, d, la, fo, trust); sd_add(T, d, la, fo, trust);
            const int wr = J.wrap[i]; if (wr >= -1024 && wr < 1024) { sd_add(&WA[wr + 1024], d, la, fo, trust); int *tq = &twr[(size_t)ti * 4]; if (!tq[1] || tq[0] == wr) { tq[0] = wr; tq[1]++; } else if (--tq[1] == 0) tq[0] = wr; }   /* (majority vote) */
        }
        for (int tr = 0; tr < th; tr++) for (int tc = 0; tc < tw; tc++) { const sd_acc *A = &TA[tr * tw + tc]; if (A->n < 0.05 * tpx * tpx) continue;
            fprintf(ft, "%s\t%d\t%d\t%.2f\t%.2f\t%d", names[s], tc, tr, tc * tilecm, tr * tilecm, twr[(size_t)(tr * tw + tc) * 4 + 1] ? twr[(size_t)(tr * tw + tc) * 4] : -9999); sd_row(ft, A, cm2px); fputc('\n', ft); }
        free(twr);
        if (SA->n > 0) { fprintf(fs, "%s\t%d\t%d", names[s], w, h); sd_row(fs, SA, cm2px); fputc('\n', fs); fflush(fs);
                         printf("  %s %dx%d: valid %.1f cm2, dist p50 %.4f p95 %.4f, trusted %.4f, folded %.5f\n", names[s], w, h, SA->n * cm2px, sd_q(SA->hd, SA->n, 0.5, 4000, 0), sd_q(SA->hd, SA->n, 0.95, 4000, 0), SA->trust / SA->n, SA->fold / SA->n); }
        free(J.dist); free(J.la); free(J.fold); free(J.wrap);
    }
    for (int k = 0; k < 2048; k++) if (WA[k].n > 0) { fprintf(fw, "w%03d", k - 1024); sd_row(fw, &WA[k], cm2px); fputc('\n', fw); }
    if (T->n > 0) printf("strip-distort: total valid %.1f cm2: dist p50 %.4f p95 %.4f p99 %.4f, area p50 %.4f, trusted (dist <= %.3f) %.4f, folded %.5f -> %s\n", T->n * cm2px,
                         sd_q(T->hd, T->n, 0.5, 4000, 0), sd_q(T->hd, T->n, 0.95, 4000, 0), sd_q(T->hd, T->n, 0.99, 4000, 0), exp2(sd_q(T->ha, T->n, 0.5, 2000, -1)), trust, T->trust / T->n, T->fold / T->n, od);
    fclose(ft); fclose(fs); fclose(fw); free(T); free(WA); free(SA); free(TA); free(names); free(segc0); free(segw);
    return 0;
}
