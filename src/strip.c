/* A whole scroll as ONE continuous surface (ufsm strip), flattened as one long sheet (ufsm strip-flatten), rendered as stripes and an
   overview (ufsm strip-mosaic).

   Why concatenating the wraps is the spiral: the fitter's winding coordinate is W = U - dir theta / 2 pi with U smooth and single valued
   (src/fit.h) and theta = atan2 in (-pi, pi]. Wrap k (ufsm wraps) is {W = k} on rays theta in [-pi, pi]: at theta -> -pi it has
   U = k - dir / 2, at theta -> +pi U = k + dir / 2. U is continuous across the cut, so wrap k's theta = -pi edge IS wrap k - dir's
   theta = +pi edge: the integer level sets are consecutive revolutions of one sheet, joined at the cut. Walking them in that order (for
   dir = +1 from the outermost wrap inwards) is walking the sheet continuously; the strip's column is the unrolled angle Theta (per wrap
   uniform in theta at that wrap's arc step, so the column pitch is about the arc step everywhere) and its row is z. ufsm wraps' column 0
   is theta = th0 + (cols - 1) dth >= pi: its ray, beyond the cut, meets the same wrap's other end one turn further in, so it is dropped
   (measured: strip prints its jump in units of the column pitch). The junction order (which wrap's -pi edge meets which wrap's +pi
   edge) is not taken from the model: both pairings are measured and the one whose edges coincide wins, which also reports the seam gaps.

   ufsm strip <wraps-dir> <out-dir> [--first K] [--last K] [--every E] [--err 0.25] [--threads 8]
     -> <out-dir>/strip.sfc (one surface-compressor file: x y z, rows z, columns Theta, joint XYZ within --err voxels, region-readable
        64 x 64 blocks: readers stream column ranges) + strip.tsv (segments: wrap, first strip column, columns, seam gap) + strip.json.
   ufsm strip-flatten <strip-dir> [--out DIR] [--decim D] [--stride S] [--iters 30] [--cg 200] [--threads 16] [--ostep S]
                      [--stripe-px 60000] [--px-scale 1] [--flip-rows 1] [--max-edge 3]
     Hierarchical flattening of the whole strip (global SLIM on ~1e9 vertices is infeasible and would not stay a straight band):
     1. the global frame: T(c, r) = (U(c), r step), U(c) the cumulative median (over rows) 3D distance between consecutive columns: an
        arc-length parametrisation shared by every row, so the strip stays one straight band of the scroll's height;
     2. chunks of 2 S columns centred every S columns (full height): each SLIM-flattened (src/flatten.h, every connected component,
        multilevel, flip-free) and every component rigidly aligned (rotation + translation, no scale) onto T;
     3. partition of unity: each lattice vertex's uv = sum over chunks of hat(c) x that chunk's uv (hats of half-width S, summing to 1):
        neighbouring chunks agree on their overlap, the residual bend of the true development is spread smoothly instead of torn. Holes
        and disconnected pieces cannot tear the layout: every piece is placed by T.
     4. distortion of T and of the result on the lattice mesh (area-weighted aniso p50 / p95, flips) overall and per stride segment;
     5. the uv map resampled on an isometric grid of pitch --ostep, written as stripes <out>/stripes/wsNNN (tifxyz + gu.tif / gv.tif =
        the strip's source column / row, as ufsm flatten) of at most --stripe-px pixels at render --scale px-scale: ufsm render
        <out>/stripes renders them (each < the JPEG limit).
   ufsm strip-mosaic <flat-dir> <render-dir> <out-dir> [--ct-level 80] [--overview-px 16000] [--rows 10] [--tile-cm 1] [--quality 88]
     stripes (CT over ink, global contrast stretch) as JPEGs, the whole unrolled scroll as a folded overview JPEG and one long strip
     JPEG, and the on-papyrus share along the strip (CT >= ct-level within 3 x 3 px around the surface point) per 1 cm column and per
     wrap (strip_papyrus.tsv, papyrus_wraps.tsv): where the surface leaves the papyrus is reported, never cut. */
#include "flatten.h"
#include "imgops.h"
#include "jpegpar.h"
#include "tiffw.h"
#include "surfcomp.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double snow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *sopt(int argc, char **argv, const char *name, const char *def) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1]; return def; }
static double meta_num(const char *m, const char *key, double def) { const char *q = m ? strstr(m, key) : nullptr; if (!q) return def; q += strlen(key); while (*q == ' ' || *q == ':' || *q == '[' || *q == '"') q++; return atof(q); }
static int cmp_f32(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static float median_f32(float *v, int n) { if (n <= 0) return NAN; qsort(v, (size_t)n, 4, cmp_f32); return v[n / 2]; }

/* ---- a static parallel for ---- */
typedef void (*sp_fn)(void *ctx, long lo, long hi);
typedef struct { sp_fn f; void *ctx; long n; int nt, t; } sp_arg;
static void *sp_run(void *a) { sp_arg *p = a; p->f(p->ctx, p->n * p->t / p->nt, p->n * (p->t + 1) / p->nt); return nullptr; }
static void spfor(long n, int nt, sp_fn f, void *ctx) {
    if (nt > 64) nt = 64;
    if (nt > n) nt = (int)n;
    if (nt <= 1) { if (n > 0) f(ctx, 0, n); return; }
    pthread_t th[64]; sp_arg a[64];
    for (int t = 0; t < nt; t++) { a[t] = (sp_arg){f, ctx, n, nt, t}; if (t) pthread_create(&th[t], nullptr, sp_run, &a[t]); }
    sp_run(&a[0]);
    for (int t = 1; t < nt; t++) pthread_join(th[t], nullptr);
}

/* ======================= ufsm strip ======================= */
typedef struct { int id, w, h; float *X, *Y, *Z; double step, um; char name[256]; } st_wrap;
static inline int vok(const st_wrap *s, size_t i) { return s->Z[i] >= 0 && isfinite(s->X[i]) && isfinite(s->Y[i]) && isfinite(s->Z[i]); }
typedef struct { const char *dir; char (*names)[256]; st_wrap *W; int n, every; _Atomic int next, fail; } st_load_job;
static void *st_load_worker(void *arg) {
    st_load_job *J = arg;
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < J->n;) {
        st_wrap *s = &J->W[k]; char p[1400]; snprintf(s->name, sizeof s->name, "%s", J->names[k]);
        snprintf(p, sizeof p, "%s/%s/meta.json", J->dir, s->name); char *m = tfx_meta_read(p);
        const double sc = meta_num(m, "\"scale\"", 1); s->um = meta_num(m, "\"voxel_size_um\"", 0); s->id = (int)meta_num(m, "\"winding\"", atoi(s->name + 1)); free(m);
        s->step = sc > 0 ? 1 / sc : 1;
        static const char *ax[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0, ok = 1;
        for (int a = 0; a < 3 && ok; a++) { int w2, h2; snprintf(p, sizeof p, "%s/%s/%s.tif", J->dir, s->name, ax[a]); A[a] = img_read_f32(p, &w2, &h2); ok = A[a] && (!a || (w2 == w && h2 == h)); w = w2; h = h2; }
        if (!ok) { fprintf(stderr, "strip: cannot read %s/%s\n", J->dir, s->name); for (int a = 0; a < 3; a++) free(A[a]); atomic_store(&J->fail, 1); continue; }
        const int E = J->every;
        if (E > 1) {   /* every E-th vertex in both directions (a step-4 set as step 8) */
            const int ow = (w - 1) / E + 1, oh = (h - 1) / E + 1;
            for (int a = 0; a < 3; a++) { float *o = malloc((size_t)ow * oh * 4); for (int r = 0; r < oh; r++) for (int c = 0; c < ow; c++) o[(size_t)r * ow + c] = A[a][(size_t)r * E * w + (size_t)c * E]; free(A[a]); A[a] = o; }
            w = ow; h = oh; s->step *= E;
        }
        s->X = A[0]; s->Y = A[1]; s->Z = A[2]; s->w = w; s->h = h;
    }
    return nullptr;
}
/* the median over rows of the 3D distance between column ca of a and column cb of b (rows valid in both); nan when < 8 rows */
static double col_dist(const st_wrap *a, int ca, const st_wrap *b, int cb, float *buf) {
    int n = 0;
    for (int r = 0; r < a->h; r++) {
        const size_t i = (size_t)r * a->w + ca, j = (size_t)r * b->w + cb; if (!vok(a, i) || !vok(b, j)) continue;
        const double dx = a->X[i] - b->X[j], dy = a->Y[i] - b->Y[j], dz = a->Z[i] - b->Z[j]; buf[n++] = (float)sqrt(dx * dx + dy * dy + dz * dz);
    }
    return n >= 8 ? median_f32(buf, n) : NAN;
}
/* the median column pitch of wrap a over columns [c0, c1) (nan without data) */
static double col_pitch(const st_wrap *a, int c0, int c1, float *buf) {
    double acc[64]; int n = 0;
    for (int c = c0; c < c1 && c + 1 < a->w && n < 64; c++) { const double d = col_dist(a, c, a, c + 1, buf); if (isfinite(d)) acc[n++] = d; }
    if (!n) return NAN;
    float v[64]; for (int k = 0; k < n; k++) v[k] = (float)acc[k];
    return median_f32(v, n);
}

int cmd_strip(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm strip <wraps-dir> <out-dir> [--first K] [--last K] [--every E] [--err 0.25] [--threads 8]\n"); return 2; }
    const char *wd = argv[2], *od = argv[3];
    const int first = atoi(sopt(argc, argv, "--first", "-1000000")), last = atoi(sopt(argc, argv, "--last", "1000000")), every = atoi(sopt(argc, argv, "--every", "1"));
    const int nth = atoi(sopt(argc, argv, "--threads", "8")); const double err = atof(sopt(argc, argv, "--err", "0.25"));
    if (every < 1 || !(err > 0)) { fprintf(stderr, "strip: --every >= 1, --err > 0\n"); return 2; }
    const double t0 = snow();
    DIR *d = opendir(wd); if (!d) { fprintf(stderr, "strip: cannot open %s\n", wd); return 1; }
    int n = 0, cap = 512; char (*names)[256] = malloc((size_t)cap * 256); struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] != 'w' || de->d_name[1] < '0' || de->d_name[1] > '9') continue;
        char nm[256]; snprintf(nm, sizeof nm, "%s", de->d_name); const size_t l = strlen(nm); if (l > 4 && !strcmp(nm + l - 4, ".sfc")) nm[l - 4] = 0;
        const int id = atoi(nm + 1); if (id < first || id > last) continue;
        int dup = 0; for (int k = 0; k < n; k++) dup |= !strcmp(names[k], nm); if (dup) continue;
        if (n == cap) { cap *= 2; names = realloc(names, (size_t)cap * 256); }
        snprintf(names[n++], 256, "%s", nm);
    }
    closedir(d);
    if (n < 1) { fprintf(stderr, "strip: no wraps in %s\n", wd); return 1; }
    /* by winding id */
    for (int a = 1; a < n; a++) for (int b = a; b > 0 && atoi(names[b] + 1) < atoi(names[b - 1] + 1); b--) { char t[256]; memcpy(t, names[b], 256); memcpy(names[b], names[b - 1], 256); memcpy(names[b - 1], t, 256); }
    st_wrap *W = calloc((size_t)n, sizeof *W);
    st_load_job LJ = {wd, names, W, n, every, 0, 0};
    { pthread_t th[64]; const int m = nth < 1 ? 1 : nth > 64 ? 64 : nth; for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, st_load_worker, &LJ); for (int t = 0; t < m; t++) pthread_join(th[t], nullptr); }
    if (LJ.fail) return 1;
    const int h = W[0].h; const double step = W[0].step, um = W[0].um;
    for (int k = 0; k < n; k++) if (W[k].h != h || fabs(W[k].step - step) > 1e-6 || fabs(W[k].um - um) > 1e-9 || W[k].w < 4) {
        fprintf(stderr, "strip: %s is %dx%d step %g um %g, %s %d rows step %g um %g: one model's wraps needed\n", W[k].name, W[k].w, W[k].h, W[k].step, W[k].um, W[0].name, h, step, um); return 1; }
    fprintf(stderr, "strip: %d wraps %s..%s, %d rows, step %g (%g um), loaded in %.1fs\n", n, W[0].name, W[n - 1].name, h, step, um, snow() - t0);
    float *buf = malloc((size_t)h * 4);
    /* column 0 (beyond the cut): its jump from column 1 vs the pitch */
    { float *jr = malloc((size_t)n * 4); int nj = 0;
      for (int k = 0; k < n; k++) { const double j0 = col_dist(&W[k], 0, &W[k], 1, buf), p = col_pitch(&W[k], 1, 9, buf); if (isfinite(j0) && p > 0) jr[nj++] = (float)(j0 / p); }
      fprintf(stderr, "strip: column 0 to 1 distance / column pitch: median %.2f over %d wraps (column 0 dropped: it lies past the cut)\n", nj ? median_f32(jr, nj) : NAN, nj); free(jr); }
    /* the junction order: A = wrap k's last column against wrap k+1's column 1 (ascending walk), B = wrap k+1's last column against wrap
       k's column 1 (descending walk); only id-consecutive pairs */
    int va = 0, vb = 0;
    for (int k = 0; k + 1 < n; k++) {
        if (W[k + 1].id != W[k].id + 1) continue;
        const double a = col_dist(&W[k], W[k].w - 1, &W[k + 1], 1, buf), b = col_dist(&W[k + 1], W[k + 1].w - 1, &W[k], 1, buf);
        if (isfinite(a) && isfinite(b)) { if (a < b) va++; else vb++; }
    }
    const int desc = vb > va;   /* descending ids: dir = +1 (outermost first) */
    fprintf(stderr, "strip: junction votes ascending %d descending %d -> walk %s (column 0 = wrap %s's first column)\n", va, vb, desc ? "outermost first (descending ids)" : "innermost first (ascending ids)", desc ? W[n - 1].name : W[0].name);
    /* the sequence: per wrap columns [1, w) (minus its last when the junction is closer than half a pitch) */
    int *ord = malloc((size_t)n * 4); for (int k = 0; k < n; k++) ord[k] = desc ? n - 1 - k : k;
    int *cb = malloc((size_t)n * 4), *ce = malloc((size_t)n * 4); double *gap = malloc((size_t)n * 8);
    for (int q = 0; q < n; q++) { cb[q] = 1; ce[q] = W[ord[q]].w; gap[q] = NAN; }
    int nbad = 0; float *gr = malloc((size_t)n * 4); int ng = 0;
    for (int q = 0; q + 1 < n; q++) {
        const st_wrap *a = &W[ord[q]], *b = &W[ord[q + 1]];
        if (abs(a->id - b->id) != 1) { fprintf(stderr, "strip: wraps %s -> %s are not consecutive: the strip has a gap there\n", a->name, b->name); continue; }
        const double p = col_pitch(a, a->w - 10, a->w - 1, buf); double g = col_dist(a, a->w - 1, b, 1, buf);
        if (isfinite(g) && p > 0 && g < 0.5 * p) { ce[q]--; g = col_dist(a, a->w - 2, b, 1, buf); }
        gap[q] = p > 0 ? g / p : NAN;
        if (isfinite(gap[q])) { gr[ng++] = (float)gap[q]; if (gap[q] > 2.5) { nbad++; fprintf(stderr, "strip: seam %s -> %s: gap %.2f column pitches\n", a->name, b->name, gap[q]); } }
    }
    fprintf(stderr, "strip: seam gaps / pitch: median %.2f over %d seams, %d above 2.5\n", ng ? median_f32(gr, ng) : NAN, ng, nbad);
    free(gr);
    long long Wt = 0; for (int q = 0; q < n; q++) Wt += ce[q] - cb[q];
    if (Wt > 0x7fffffffLL) { fprintf(stderr, "strip: %lld columns: too many\n", Wt); return 1; }
    const int w = (int)Wt;
    int *mw = malloc((size_t)w * 4), *mc = malloc((size_t)w * 4);
    { int c = 0; for (int q = 0; q < n; q++) for (int k = cb[q]; k < ce[q]; k++) { mw[c] = ord[q]; mc[c] = k; c++; } }
    char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", od); if (system(cmd)) return 1;
    char p[1500];
    snprintf(p, sizeof p, "%s/strip.tsv", od); FILE *tf = fopen(p, "w"); if (!tf) return 1;
    fprintf(tf, "seg\twrap\tname\tcol0\tncols\twrap_col0\tseam_gap_pitch\n");
    { long long c = 0; for (int q = 0; q < n; q++) { fprintf(tf, "%d\t%d\t%s\t%lld\t%d\t%d\t%.3f\n", q, W[ord[q]].id, W[ord[q]].name, c, ce[q] - cb[q], cb[q], gap[q]); c += ce[q] - cb[q]; } }
    fclose(tf);
    /* strip.sfc, written block row by block row */
    size_t nvalid = 0; double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int c = 0; c < w; c++) { const st_wrap *s = &W[mw[c]]; for (int r = 0; r < h; r++) { const size_t i = (size_t)r * s->w + mc[c]; if (!vok(s, i)) continue; nvalid++;
        const float v[3] = {s->X[i], s->Y[i], s->Z[i]}; for (int a = 0; a < 3; a++) { if (v[a] < lo[a]) lo[a] = v[a]; if (v[a] > hi[a]) hi[a] = v[a]; } } }
    char meta[2048];
    snprintf(meta, sizeof meta, "{\"format\": \"tifxyz\", \"type\": \"strip\", \"uuid\": \"strip\", \"scale\": [%.9g, %.9g], \"bbox\": [[%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f]], "
             "\"voxel_size_um\": %.6g, \"winding\": -1, \"wraps\": [%d, %d], \"walk\": \"%s\", \"valid_vertices\": %zu, \"source\": \"ufsm strip (consecutive wraps joined at the theta cut)\"}\n",
             1 / step, 1 / step, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], um, W[ord[0]].id, W[ord[n - 1]].id, desc ? "descending" : "ascending", nvalid);
    snprintf(p, sizeof p, "%s/strip.json", od); { FILE *f = fopen(p, "w"); if (f) { fputs(meta, f); fclose(f); } }
    sfc_channel dc[3]; memset(dc, 0, sizeof dc); static const char *cn[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; a++) { snprintf(dc[a].name, sizeof dc[a].name, "%s", cn[a]); dc[a].width = (uint64_t)w; dc[a].height = (uint64_t)h; dc[a].dtype = SFC_F32;
                                  dc[a].flags = SFC_COORDINATE | SFC_XYZ; dc[a].components = 3; dc[a].component = (uint32_t)a; dc[a].tolerance = err; }
    char sp[1500], tmp[1600]; snprintf(sp, sizeof sp, "%s/strip.sfc", od); snprintf(tmp, sizeof tmp, "%s.tmp%d", sp, (int)getpid()); unlink(tmp);
    sfc_writer *wr = nullptr; if (sfc_create(tmp, dc, 3, meta, strlen(meta), &wr)) { fprintf(stderr, "strip: cannot create %s\n", tmp); return 1; }
    float (*joint)[3] = malloc(4096 * 12); uint8_t *valid = malloc(4096); int rc = 0; const double tw = snow();
    for (int by = 0; by < (h + 63) / 64 && !rc; by++) for (int bx = 0; bx < (w + 63) / 64 && !rc; bx++) {
        memset(joint, 0, 4096 * 12); memset(valid, 0, 4096);
        for (int q = 0; q < 4096; q++) {
            const int c = bx * 64 + q % 64, r = by * 64 + q / 64; if (c >= w || r >= h) continue;
            const st_wrap *s = &W[mw[c]]; const size_t i = (size_t)r * s->w + mc[c];
            if (!vok(s, i) || !(s->Z[i] > 0)) continue;
            joint[q][0] = s->X[i]; joint[q][1] = s->Y[i]; joint[q][2] = s->Z[i]; valid[q] = 1;
        }
        rc = sfc_write_xyz_block(wr, joint, 12, 768, valid);
    }
    free(joint); free(valid);
    if (rc) { sfc_cancel(wr); fprintf(stderr, "strip: sfc write failed\n"); return 1; }
    if (sfc_finish(wr) || rename(tmp, sp)) { unlink(tmp); fprintf(stderr, "strip: sfc finish failed\n"); return 1; }
    struct stat st; stat(sp, &st);
    double pmed = 0; { float *pv = malloc((size_t)n * 4); int np = 0; for (int k = 0; k < n; k++) { const double pk = col_pitch(&W[k], W[k].w / 2, W[k].w / 2 + 16, buf); if (pk > 0) pv[np++] = (float)pk; } pmed = np ? median_f32(pv, np) : step; free(pv); }
    printf("strip: %d wraps -> %s: %d x %d (columns x rows), %zu valid vertices, column pitch median %.2f voxels (step %g), length ~%.1f cm, %.1f MB, write %.1fs, total %.1fs\n",
           n, sp, w, h, nvalid, pmed, step, w * pmed * um * 1e-4, st.st_size / 1e6, snow() - tw, snow() - t0);
    for (int k = 0; k < n; k++) { free(W[k].X); free(W[k].Y); free(W[k].Z); }
    free(W); free(names); free(buf); free(ord); free(cb); free(ce); free(gap); free(mw); free(mc);
    return 0;
}

/* ======================= reading a strip ======================= */
typedef struct { sfc_reader *r; int cx, w, h; double step, um; char path[1400]; } st_strip;
static int st_open(const char *dir, st_strip *S) {
    memset(S, 0, sizeof *S); snprintf(S->path, sizeof S->path, "%s/strip.sfc", dir);
    if (sfc_open_file(S->path, &S->r)) { fprintf(stderr, "strip: cannot open %s\n", S->path); return -1; }
    S->cx = sfc_find_channel(S->r, "x"); if (S->cx < 0) S->cx = 0;
    const sfc_channel *c = sfc_channel_info(S->r, (uint32_t)S->cx); S->w = (int)c->width; S->h = (int)c->height;
    const uint64_t ml = sfc_metadata_size(S->r); char *m = calloc(ml + 1, 1); if (ml) sfc_read_metadata(S->r, m, ml);
    const double sc = meta_num(m, "\"scale\"", 1); S->step = sc > 0 ? 1 / sc : 1; S->um = meta_num(m, "\"voxel_size_um\"", 1); free(m);
    return 0;
}
static void st_close(st_strip *S) { if (S->r) sfc_close(S->r); S->r = nullptr; }
typedef struct { st_strip *S; int c0, nc; float *X, *Y, *Z; _Atomic int next, fail; } st_read_job;
static void *st_read_worker(void *arg) {
    st_read_job *J = arg; sfc_cache *ca = nullptr; if (sfc_cache_create(J->S->r, (size_t)64 << 20, &ca)) { atomic_store(&J->fail, 1); return nullptr; }
    float *xyz = malloc((size_t)J->nc * 64 * 12); uint8_t *v = malloc((size_t)J->nc * 64);
    for (int b; (b = atomic_fetch_add(&J->next, 1)) * 64 < J->S->h;) {
        const int r0 = b * 64, nr = J->S->h - r0 < 64 ? J->S->h - r0 : 64;
        if (sfc_cache_read_xyz_region(ca, (uint32_t)J->S->cx, (uint64_t)J->c0, (uint64_t)r0, (uint32_t)J->nc, (uint32_t)nr, xyz, 12, (size_t)J->nc * 12, v)) { atomic_store(&J->fail, 1); break; }
        for (int r = 0; r < nr; r++) for (int c = 0; c < J->nc; c++) {
            const size_t i = (size_t)(r0 + r) * J->nc + c, q = (size_t)r * J->nc + c;
            if (v[q]) { J->X[i] = xyz[3 * q]; J->Y[i] = xyz[3 * q + 1]; J->Z[i] = xyz[3 * q + 2]; } else J->X[i] = J->Y[i] = J->Z[i] = -1.f;
        }
        sfc_cache_clear(ca);
    }
    free(xyz); free(v); sfc_cache_destroy(ca);
    return nullptr;
}
/* columns [c0, c0 + nc) x all rows into X / Y / Z (h x nc, row-major; invalid -1), decoded by 64-row bands on nth threads */
static int st_read(st_strip *S, int c0, int nc, int nth, float *X, float *Y, float *Z) {
    st_read_job J = {S, c0, nc, X, Y, Z, 0, 0};
    pthread_t th[64]; const int m = nth < 1 ? 1 : nth > 64 ? 64 : nth;
    for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, st_read_worker, &J);
    for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
    return J.fail ? -1 : 0;
}
typedef struct { int wrap; long long c0; int n; } st_seg;
static int st_segs(const char *dir, st_seg **out) {   /* strip.tsv */
    char p[1400]; snprintf(p, sizeof p, "%s/strip.tsv", dir); FILE *f = fopen(p, "r"); if (!f) return -1;
    char line[1024]; int n = 0, cap = 256; st_seg *s = malloc((size_t)cap * sizeof *s);
    if (!fgets(line, sizeof line, f)) { fclose(f); free(s); return -1; }
    while (fgets(line, sizeof line, f)) {
        int q, wr, nc, wc; long long c0; char nm[256];
        if (sscanf(line, "%d\t%d\t%255s\t%lld\t%d\t%d", &q, &wr, nm, &c0, &nc, &wc) != 6) continue;
        if (n == cap) { cap *= 2; s = realloc(s, (size_t)cap * sizeof *s); }
        s[n++] = (st_seg){wr, c0, nc};
    }
    fclose(f); *out = s; return n;
}
static int seg_wrap(const st_seg *s, int n, double col) {   /* the wrap of a strip column */
    int lo = 0, hi = n - 1; if (n <= 0) return -1;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (s[mid].c0 <= col) lo = mid; else hi = mid - 1; }
    return s[lo].wrap;
}

/* ======================= distortion on the lattice ======================= */
/* the Jacobian of the map from a 3D triangle (its own 2D frame) to uv: singular values s1 >= s2, det sign */
static int tri_sv(const double p[3][3], const double uv[3][2], double *s1, double *s2, double *area, double *det) {
    double e1[3], e2[3]; for (int d = 0; d < 3; d++) { e1[d] = p[1][d] - p[0][d]; e2[d] = p[2][d] - p[0][d]; }
    const double l1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    const double nn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double nl = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]); if (!(l1 > 0) || !(nl > 0)) return -1;
    const double qx = (e2[0] * e1[0] + e2[1] * e1[1] + e2[2] * e1[2]) / l1, qy = nl / l1;
    /* frame coords: p0 (0,0), p1 (l1, 0), p2 (qx, qy); J = [du dv] [frame]^-1 */
    const double a = uv[1][0] - uv[0][0], b = uv[2][0] - uv[0][0], c = uv[1][1] - uv[0][1], e = uv[2][1] - uv[0][1];
    const double J0 = a / l1, J1 = (b - J0 * qx) / qy, J2 = c / l1, J3 = (e - J2 * qx) / qy;
    *det = J0 * J3 - J1 * J2;
    const double E = (J0 + J3) / 2, F = (J0 - J3) / 2, G = (J2 + J1) / 2, H = (J2 - J1) / 2, Q = hypot(E, H), R = hypot(F, G);
    *s1 = Q + R; *s2 = Q - R; *area = 0.5 * nl; return 0;
}
enum { HB = 2000 };   /* aniso histogram: log(s1/s2) in [0, 2) over HB bins */
typedef struct { double area, flips_area, energy, a12; double h[HB + 1]; long long nt, flips; } dstat;
static void dstat_add(dstat *s, double s1, double s2, double area, double det) {
    s->nt++; s->area += area;
    if (!(det > 0) || !(s2 > 0)) { s->flips++; s->flips_area += area; return; }
    const double r = s1 / s2; int b = (int)(log(r) / 2.0 * HB); if (b < 0) b = 0; if (b > HB) b = HB; s->h[b] += area;
    if (r > 1.2) s->a12 += area;
    s->energy += area * (s1 * s1 + 1 / (s1 * s1) + s2 * s2 + 1 / (s2 * s2));
}
static double dstat_q(const dstat *s, double q) {
    double tot = 0; for (int b = 0; b <= HB; b++) tot += s->h[b]; if (!(tot > 0)) return NAN;
    double c = 0; for (int b = 0; b <= HB; b++) { c += s->h[b]; if (c >= q * tot) return exp((b + 0.5) * 2.0 / HB); }
    return exp(2.0);
}
static void dstat_print(FILE *f, const char *label, const dstat *s) {
    const double ok = s->area - s->flips_area;
    fprintf(f, "%s: aniso p50/p95/p99 %.3f/%.3f/%.3f aniso>1.2 %.2f%% energy %.4f flips %lld (%.4f%% area) tris %lld\n", label, dstat_q(s, 0.5), dstat_q(s, 0.95), dstat_q(s, 0.99),
            ok > 0 ? 100 * s->a12 / ok : 0, ok > 0 ? s->energy / ok : 0, s->flips, s->area > 0 ? 100 * s->flips_area / s->area : 0, s->nt);
}

/* ======================= ufsm strip-flatten ======================= */
typedef struct {
    st_strip S; int w, h, D, lw, lh, Sst; double step, max_edge;
    double *U;              /* per strip column: the frame's arc length (voxels) */
    float *du, *dv, *wt;    /* per lattice vertex: the blended uv minus the frame T, and the weight sum (after: du / dv final, wt 1 or 0) */
} sflat;
static inline double T_u(const sflat *F, int c) { return F->U[c]; }
static inline double T_v(const sflat *F, int r) { return r * F->step; }

typedef struct { const float *X, *Y, *Z; int nc, h; float *med; } cpj;
static void k_colpitch(void *ctx, long lo, long hi) {
    cpj *J = ctx; float *buf = malloc((size_t)J->h * 4);
    for (long c = lo; c < hi; c++) {
        int n = 0;
        for (int r = 0; r < J->h; r++) {
            const size_t i = (size_t)r * J->nc + c; if (J->Z[i] < 0 || J->Z[i + 1] < 0) continue;
            const double dx = J->X[i + 1] - J->X[i], dy = J->Y[i + 1] - J->Y[i], dz = J->Z[i + 1] - J->Z[i]; buf[n++] = (float)sqrt(dx * dx + dy * dy + dz * dz);
        }
        J->med[c] = n >= 1 ? median_f32(buf, n) : NAN;
    }
    free(buf);
}

/* per chunk: the components of its mesh, each rigidly aligned onto the frame T */
static void align_components(const sflat *F, flat_mesh *m, int c0, int nc) {
    int *par = malloc((size_t)m->nv * 4); for (int v = 0; v < m->nv; v++) par[v] = v;
    for (int t = 0; t < m->nt; t++) for (int k = 1; k < 3; k++) {
        int a = m->tri[3 * t], b = m->tri[3 * t + k];
        while (par[a] != a) a = par[a] = par[par[a]];
        while (par[b] != b) b = par[b] = par[par[b]];
        if (a != b) par[b] = a;
    }
    for (int v = 0; v < m->nv; v++) { int a = v; while (par[a] != a) a = par[a]; par[v] = a; }
    /* per root: centroids and cross terms */
    double *acc = calloc((size_t)m->nv * 7, 8);   /* n, su, sv, st, sw, (sum later) */
    for (int v = 0; v < m->nv; v++) {
        const int g = m->gi[v], c = g % nc, r = g / nc; double *a = acc + 7 * (size_t)par[v];
        a[0] += 1; a[1] += m->uv[2 * v]; a[2] += m->uv[2 * v + 1]; a[3] += T_u(F, c0 + c); a[4] += T_v(F, r);
    }
    for (int v = 0; v < m->nv; v++) {
        const int g = m->gi[v], c = g % nc, r = g / nc; double *a = acc + 7 * (size_t)par[v];
        const double ax = m->uv[2 * v] - a[1] / a[0], ay = m->uv[2 * v + 1] - a[2] / a[0], bx = T_u(F, c0 + c) - a[3] / a[0], by = T_v(F, r) - a[4] / a[0];
        a[5] += ax * bx + ay * by; a[6] += ax * by - ay * bx;
    }
    for (int v = 0; v < m->nv; v++) {
        const double *a = acc + 7 * (size_t)par[v]; const double th = atan2(a[6], a[5]), cs = cos(th), sn = sin(th);
        const double ax = m->uv[2 * v] - a[1] / a[0], ay = m->uv[2 * v + 1] - a[2] / a[0];
        m->uv[2 * v] = cs * ax - sn * ay + a[3] / a[0]; m->uv[2 * v + 1] = sn * ax + cs * ay + a[4] / a[0];
    }
    free(par); free(acc);
}

/* the lattice mesh's distortion over lattice columns [l0, l1), T (frame) and the blended map; per segment of segl lattice columns */
typedef struct { const sflat *F; int l0, l1; const float *X, *Y, *Z; int c0, nc; dstat *before, *after; int segl, seg0; dstat *segs; } dsj;
static void lattice_stats(const sflat *F, const float *X, const float *Y, const float *Z, int c0, int nc, int l0, int l1, dstat *before, dstat *after, dstat *segs, int segl) {
    const int D = F->D, lw = F->lw; const double sp = D * F->step, lmax2 = (F->max_edge * sp * 1.4143) * (F->max_edge * sp * 1.4143), amin = 1e-4 * sp * sp;
    for (int i = 0; i + 1 < F->lh; i++) for (int j = l0; j < l1 && j + 1 < lw; j++) {
        const int LC[4][2] = {{i, j}, {i, j + 1}, {i + 1, j}, {i + 1, j + 1}}; int ok[4], nok = 0; size_t li[4], gi[4];
        for (int k = 0; k < 4; k++) {
            li[k] = (size_t)LC[k][0] * lw + LC[k][1]; const int c = LC[k][1] * D - c0, r = LC[k][0] * D; gi[k] = (size_t)r * nc + c;
            ok[k] = c >= 0 && c < nc && Z[gi[k]] >= 0 && F->wt[li[k]] > 0; nok += ok[k];
        }
        int cand[2][3], ncnd = 0;
        if (nok == 4) { cand[0][0] = 0; cand[0][1] = 1; cand[0][2] = 3; cand[1][0] = 0; cand[1][1] = 3; cand[1][2] = 2; ncnd = 2; }
        else if (nok == 3) { static const int one[4][3] = {{1, 3, 2}, {0, 3, 2}, {0, 1, 3}, {0, 1, 2}}; for (int k = 0; k < 4; k++) if (!ok[k]) { memcpy(cand[0], one[k], sizeof cand[0]); ncnd = 1; } }
        for (int q = 0; q < ncnd; q++) {
            double p[3][3], u0[3][2], u1[3][2]; int bad = 0;
            for (int k = 0; k < 3; k++) {
                const int v = cand[q][k]; p[k][0] = X[gi[v]]; p[k][1] = Y[gi[v]]; p[k][2] = Z[gi[v]];
                u0[k][0] = T_u(F, LC[v][1] * D); u0[k][1] = T_v(F, LC[v][0] * D); u1[k][0] = u0[k][0] + F->du[li[v]]; u1[k][1] = u0[k][1] + F->dv[li[v]];
            }
            for (int k = 0; k < 3 && !bad; k++) { const double *a = p[k], *b = p[(k + 1) % 3]; bad = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]) > lmax2; }
            if (bad) continue;
            double s1, s2, ar, det;
            if (tri_sv(p, u0, &s1, &s2, &ar, &det) || ar < amin) continue;
            dstat_add(before, s1, s2, ar, det);
            if (tri_sv(p, u1, &s1, &s2, &ar, &det)) continue;
            dstat_add(after, s1, s2, ar, det);
            if (segs) dstat_add(&segs[j / segl], s1, s2, ar, det);
        }
    }
}

typedef struct {   /* stripe rasterisation */
    const sflat *F; const float *X, *Y, *Z; int c0, nc;   /* the full-res region */
    int la, lb;     /* lattice columns */
    double u0, v0, os; int ow, oh;
    float *gu, *gv, *ox, *oy, *oz;
} rsj;
static void k_rast(void *ctx, long lo, long hi) {
    rsj *R = ctx; const sflat *F = R->F; const int D = F->D, lw = F->lw, nc = R->nc;
    const double sp = D * F->step, lmax2 = (F->max_edge * sp * 1.4143) * (F->max_edge * sp * 1.4143);
    for (long q = lo; q < hi; q++) {
        const int i = (int)(q / (R->lb - R->la)), j = R->la + (int)(q % (R->lb - R->la)); if (i + 1 >= F->lh || j + 1 >= lw) continue;
        const int LC[4][2] = {{i, j}, {i, j + 1}, {i + 1, j}, {i + 1, j + 1}}; int ok[4], nok = 0; size_t li[4], gi[4];
        for (int k = 0; k < 4; k++) {
            li[k] = (size_t)LC[k][0] * lw + LC[k][1]; const int c = LC[k][1] * D - R->c0, r = LC[k][0] * D; gi[k] = (size_t)r * nc + c;
            ok[k] = c >= 0 && c < nc && R->Z[gi[k]] >= 0 && F->wt[li[k]] > 0; nok += ok[k];
        }
        int cand[2][3], ncnd = 0;
        if (nok == 4) { cand[0][0] = 0; cand[0][1] = 1; cand[0][2] = 3; cand[1][0] = 0; cand[1][1] = 3; cand[1][2] = 2; ncnd = 2; }
        else if (nok == 3) { static const int one[4][3] = {{1, 3, 2}, {0, 3, 2}, {0, 1, 3}, {0, 1, 2}}; for (int k = 0; k < 4; k++) if (!ok[k]) { memcpy(cand[0], one[k], sizeof cand[0]); ncnd = 1; } }
        for (int t = 0; t < ncnd; t++) {
            double U[3], V[3], P[3][3], GC[3], GR[3]; int bad = 0;
            for (int k = 0; k < 3; k++) {
                const int v = cand[t][k]; const int c = LC[v][1] * D, r = LC[v][0] * D;
                U[k] = (T_u(F, c) + F->du[li[v]] - R->u0) / R->os; V[k] = (T_v(F, r) + F->dv[li[v]] - R->v0) / R->os;
                P[k][0] = R->X[gi[v]]; P[k][1] = R->Y[gi[v]]; P[k][2] = R->Z[gi[v]]; GC[k] = c; GR[k] = r;
            }
            for (int k = 0; k < 3 && !bad; k++) { const double *a = P[k], *b = P[(k + 1) % 3]; bad = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]) > lmax2; }
            if (bad) continue;
            const double det = (U[1] - U[0]) * (V[2] - V[0]) - (V[1] - V[0]) * (U[2] - U[0]); if (!(det > 1e-12)) continue;   /* (flipped / degenerate: left out) */
            const int x0 = (int)ceil(fmin(U[0], fmin(U[1], U[2])) - 1e-9), x1 = (int)floor(fmax(U[0], fmax(U[1], U[2])) + 1e-9);
            const int y0 = (int)ceil(fmin(V[0], fmin(V[1], V[2])) - 1e-9), y1 = (int)floor(fmax(V[0], fmax(V[1], V[2])) + 1e-9);
            for (int y = y0 < 0 ? 0 : y0; y <= y1 && y < R->oh; y++) for (int x = x0 < 0 ? 0 : x0; x <= x1 && x < R->ow; x++) {
                const double l1 = ((x - U[0]) * (V[2] - V[0]) - (y - V[0]) * (U[2] - U[0])) / det, l2 = ((U[1] - U[0]) * (y - V[0]) - (V[1] - V[0]) * (x - U[0])) / det, l0 = 1 - l1 - l2;
                if (l0 < -1e-7 || l1 < -1e-7 || l2 < -1e-7) continue;
                const double L[3] = {l0, l1, l2}; double gc = 0, gr = 0, p[3] = {0, 0, 0};
                for (int k = 0; k < 3; k++) { gc += L[k] * GC[k]; gr += L[k] * GR[k]; for (int d = 0; d < 3; d++) p[d] += L[k] * P[k][d]; }
                /* the full-res grid's bilinear surface where its 4 corners are valid (as ufsm render samples a grid) */
                int cc = (int)floor(gc) - R->c0, rr = (int)floor(gr);
                if (cc > nc - 2) cc = nc - 2;
                if (rr > F->h - 2) rr = F->h - 2;
                if (cc >= 0 && rr >= 0) {
                    const size_t g = (size_t)rr * nc + cc; const double fu = gc - R->c0 - cc, fv = gr - rr;
                    if (R->Z[g] >= 0 && R->Z[g + 1] >= 0 && R->Z[g + nc] >= 0 && R->Z[g + nc + 1] >= 0) {
                        const float *A[3] = {R->X, R->Y, R->Z};
                        for (int d = 0; d < 3; d++) p[d] = (1 - fv) * ((1 - fu) * A[d][g] + fu * A[d][g + 1]) + fv * ((1 - fu) * A[d][g + nc] + fu * A[d][g + nc + 1]);
                    }
                }
                const size_t o = (size_t)y * R->ow + x;
                R->gu[o] = (float)gc; R->gv[o] = (float)gr; R->ox[o] = (float)p[0]; R->oy[o] = (float)p[1]; R->oz[o] = (float)p[2];
            }
        }
    }
}

int cmd_strip_flatten(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm strip-flatten <strip-dir> [--out DIR] [--decim D] [--stride S] [--iters 30] [--cg 200] [--threads 16] [--ostep S] [--stripe-px 60000] [--px-scale 1] [--flip-rows 1] [--max-edge 3] [--reuse 1]\n"); return 2; }
    const char *sd = argv[2]; char od[1400]; snprintf(od, sizeof od, "%s", sopt(argc, argv, "--out", "")); if (!*od) snprintf(od, sizeof od, "%s/flat", sd);
    sflat F; memset(&F, 0, sizeof F);
    if (st_open(sd, &F.S)) return 1;
    F.w = F.S.w; F.h = F.S.h; F.step = F.S.step; F.D = atoi(sopt(argc, argv, "--decim", "1")); if (F.D < 1) F.D = 1;
    F.max_edge = atof(sopt(argc, argv, "--max-edge", "3"));
    const int nth = atoi(sopt(argc, argv, "--threads", "16"));
    int S = atoi(sopt(argc, argv, "--stride", "0")); if (S <= 0) S = F.h / 2; S = S / F.D * F.D; if (S < 4 * F.D) S = 4 * F.D; F.Sst = S;
    F.lw = (F.w - 1) / F.D + 1; F.lh = (F.h - 1) / F.D + 1;
    const double um = F.S.um, ostep = atof(sopt(argc, argv, "--ostep", "0")) > 0 ? atof(sopt(argc, argv, "--ostep", "0")) : F.step;
    const double pxs = atof(sopt(argc, argv, "--px-scale", "1")); const int stripe_px = atoi(sopt(argc, argv, "--stripe-px", "60000")), flip = atoi(sopt(argc, argv, "--flip-rows", "0"));
    char cmd[1500], p[1600]; snprintf(cmd, sizeof cmd, "mkdir -p '%s/stripes'", od); if (system(cmd)) return 1;
    st_seg *segs = nullptr; const int nseg = st_segs(sd, &segs);
    const double t0 = snow();
    fprintf(stderr, "strip-flatten: %s %d x %d step %g (%g um), lattice every %d -> %d x %d, chunks of %d columns every %d\n", F.S.path, F.w, F.h, F.step, um, F.D, F.lw, F.lh, 2 * S, S);
    /* 1. the frame U(c) */
    F.U = malloc(((size_t)F.w + 1) * 8); float *med = malloc((size_t)F.w * 4);
    const int band = 8192;
    for (int c0 = 0; c0 < F.w - 1; c0 += band) {
        const int nc = c0 + band + 1 < F.w ? band + 1 : F.w - c0; const size_t n = (size_t)nc * F.h;
        float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4);
        if (!X || !Y || !Z || st_read(&F.S, c0, nc, nth, X, Y, Z)) { fprintf(stderr, "strip-flatten: read failed\n"); return 1; }
        cpj J = {X, Y, Z, nc, F.h, med + c0}; spfor(nc - 1, nth, k_colpitch, &J);
        free(X); free(Y); free(Z);
    }
    { long long nb = 0; F.U[0] = 0; for (int c = 0; c + 1 < F.w; c++) { double d = med[c]; if (!isfinite(d) || !(d > 0)) { d = 0.02 * F.step; nb++; }   /* (no surface between them: a gap of 0.02 step, so empty runs of spurious outer wraps take no length) */ else if (d > 3 * F.step) d = 3 * F.step; F.U[c + 1] = F.U[c] + d; }
      fprintf(stderr, "strip-flatten: frame: length %.2f cm (%lld column pairs without data, 0.02 step each), %.1fs\n", F.U[F.w - 1] * um * 1e-4, nb, snow() - t0); }
    free(med);
    /* 2. chunks */
    const size_t ln = (size_t)F.lw * F.lh;
    F.du = calloc(ln, 4); F.dv = calloc(ln, 4); F.wt = calloc(ln, 4);
    if (!F.du || !F.dv || !F.wt) { fprintf(stderr, "strip-flatten: out of memory for the lattice\n"); return 1; }
    char lp[1600]; snprintf(lp, sizeof lp, "%s/lattice.f32", od);
    int reused = 0;
    if (atoi(sopt(argc, argv, "--reuse", "0"))) {   /* the chunks' blended lattice of an earlier run (output stage only) */
        FILE *f = fopen(lp, "rb");
        if (f && fread(F.du, 4, ln, f) == ln && fread(F.dv, 4, ln, f) == ln && fread(F.wt, 4, ln, f) == ln) reused = 1;
        if (f) fclose(f);
        fprintf(stderr, "strip-flatten: %s %s\n", lp, reused ? "reused" : "not readable: recomputing");
    }
    const int nchunk = (F.w - 1 + S - 1) / S + 1;
    snprintf(p, sizeof p, "%s/chunks.tsv", od); FILE *cf = reused ? nullptr : fopen(p, "w");
    if (cf) fprintf(cf, "chunk\tcol0\tncols\tverts\ttris\tbefore_aniso_p50\tbefore_aniso_p95\tafter_aniso_p50\tafter_aniso_p95\tflips\tseconds\n");
    for (int j = 0; j < nchunk && !reused; j++) {
        const double tc = snow();
        const int C = j * S, a = C - S < 0 ? 0 : C - S, b = C + S > F.w - 1 ? F.w - 1 : C + S, nc = b - a + 1;
        if (nc < 4) continue;
        const size_t n = (size_t)nc * F.h; float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4);
        if (!X || !Y || !Z || st_read(&F.S, a, nc, nth, X, Y, Z)) { fprintf(stderr, "strip-flatten: read failed\n"); return 1; }
        flat_opts o = flat_opts_default(); o.decim = F.D; o.iters = atoi(sopt(argc, argv, "--iters", "30")); o.cg_iters = atoi(sopt(argc, argv, "--cg", "200"));
        o.nthreads = nth; o.max_edge = F.max_edge; o.all_components = 1; o.verbose = atoi(sopt(argc, argv, "--verbose", "0")); o.tol = atof(sopt(argc, argv, "--tol", "1e-5"));
        double *cu = malloc((size_t)nc * 8); for (int c = 0; c < nc; c++) cu[c] = T_u(&F, a + c) - T_u(&F, a); o.col_u = cu;   /* SLIM from the frame */
        flat_mesh m; memset(&m, 0, sizeof m); flat_stats sb, sa;
        if (flat_grid(X, Y, Z, nc, F.h, F.step, &o, &m, &sb, &sa)) { fprintf(stderr, "strip-flatten: chunk %d (columns %d..%d): no mesh\n", j, a, b); free(X); free(Y); free(Z); free(cu); continue; }
        align_components(&F, &m, a, nc);
        for (int v = 0; v < m.nv; v++) {
            const int g = m.gi[v], c = g % nc, r = g / nc, gc = a + c; if (gc % F.D || r % F.D) continue;
            const double wgt = fmax(0, 1 - fabs((double)gc - C) / S); if (!(wgt > 0)) continue;
            const size_t li = (size_t)(r / F.D) * F.lw + gc / F.D;
            F.du[li] += (float)(wgt * (m.uv[2 * v] - T_u(&F, gc))); F.dv[li] += (float)(wgt * (m.uv[2 * v + 1] - T_v(&F, r))); F.wt[li] += (float)wgt;
        }
        fprintf(stderr, "strip-flatten: chunk %d/%d columns %d..%d: %d vertices %d triangles, aniso p50/p95 %.3f/%.3f -> %.3f/%.3f, flips %lld, %.1fs\n",
                j + 1, nchunk, a, b, m.nv, m.nt, sb.aniso_p50, sb.aniso_p95, sa.aniso_p50, sa.aniso_p95, sa.flips, snow() - tc);
        if (cf) { fprintf(cf, "%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%lld\t%.1f\n", j, a, nc, m.nv, m.nt, sb.aniso_p50, sb.aniso_p95, sa.aniso_p50, sa.aniso_p95, sa.flips, snow() - tc); fflush(cf); }
        flat_mesh_free(&m); free(X); free(Y); free(Z); free(cu);
    }
    if (cf) fclose(cf);
    if (!reused) {
        for (size_t i = 0; i < ln; i++) if (F.wt[i] > 1e-6f) { F.du[i] /= F.wt[i]; F.dv[i] /= F.wt[i]; F.wt[i] = 1; } else { F.du[i] = F.dv[i] = 0; F.wt[i] = 0; }
        FILE *f = fopen(lp, "wb"); if (!f || fwrite(F.du, 4, ln, f) != ln || fwrite(F.dv, 4, ln, f) != ln || fwrite(F.wt, 4, ln, f) != ln) fprintf(stderr, "strip-flatten: cannot write %s\n", lp);
        if (f) fclose(f);
    }
    fprintf(stderr, "strip-flatten: chunks done, %.1fs\n", snow() - t0);
    /* 3. distortion (frame T and the result) on the lattice mesh, per segment of S columns; the uv extent per lattice column */
    const int segl = S / F.D; const int nsg = (F.lw + segl - 1) / segl;
    dstat *before = calloc(1, sizeof(dstat)), *after = calloc(1, sizeof(dstat)), *sg = calloc((size_t)nsg, sizeof(dstat));
    float *cumin = malloc((size_t)F.lw * 4), *cumax = malloc((size_t)F.lw * 4); double umin = INFINITY, umax = -INFINITY, vmin = INFINITY, vmax = -INFINITY;
    for (int j = 0; j < F.lw; j++) {
        float lo = INFINITY, hi = -INFINITY;
        for (int i = 0; i < F.lh; i++) { const size_t li = (size_t)i * F.lw + j; if (!(F.wt[li] > 0)) continue;
            const double u = T_u(&F, j * F.D) + F.du[li], v = T_v(&F, i * F.D) + F.dv[li]; if (u < lo) lo = (float)u; if (u > hi) hi = (float)u; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
        cumin[j] = lo; cumax[j] = hi; if (lo < umin) umin = lo; if (hi > umax) umax = hi;
    }
    {
        const int lband = 4096;   /* lattice columns per read */
        for (int l0 = 0; l0 < F.lw - 1; l0 += lband) {
            const int l1 = l0 + lband < F.lw - 1 ? l0 + lband : F.lw - 1, c0 = l0 * F.D, c1 = (l1 * F.D < F.w - 1 ? l1 * F.D : F.w - 1), nc = c1 - c0 + 1;
            const size_t n = (size_t)nc * F.h; float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4);
            if (!X || !Y || !Z || st_read(&F.S, c0, nc, nth, X, Y, Z)) { fprintf(stderr, "strip-flatten: read failed\n"); return 1; }
            lattice_stats(&F, X, Y, Z, c0, nc, l0, l1, before, after, sg, segl);
            free(X); free(Y); free(Z);
        }
    }
    const double L_cm = (umax - umin) * um * 1e-4, H_cm = (vmax - vmin) * um * 1e-4, A_cm2 = after->area * um * um * 1e-8;
    printf("strip-flatten: strip %d x %d step %g, flattened length %.2f cm x height %.2f cm, mesh area %.2f cm2\n", F.w, F.h, F.step, L_cm, H_cm, A_cm2);
    dstat_print(stdout, "strip-flatten: frame T (arc-length columns x z) ", before);
    dstat_print(stdout, "strip-flatten: flattened (chunk SLIM + blend)", after);
    fflush(stdout);
    snprintf(p, sizeof p, "%s/segments.tsv", od); FILE *gf = fopen(p, "w");
    if (gf) {
        fprintf(gf, "seg\tcol0\tcol1\tu_cm\twrap0\twrap1\tarea_cm2\taniso_p50\taniso_p95\tflips\n");
        for (int s = 0; s < nsg; s++) {
            const int c0 = s * segl * F.D, c1 = ((s + 1) * segl * F.D < F.w ? (s + 1) * segl * F.D : F.w) - 1;
            fprintf(gf, "%d\t%d\t%d\t%.2f\t%d\t%d\t%.3f\t%.4f\t%.4f\t%lld\n", s, c0, c1, (T_u(&F, c0) - umin) * um * 1e-4, nseg > 0 ? seg_wrap(segs, nseg, c0) : -1, nseg > 0 ? seg_wrap(segs, nseg, c1) : -1,
                    sg[s].area * um * um * 1e-8, dstat_q(&sg[s], 0.5), dstat_q(&sg[s], 0.95), sg[s].flips);
        }
        fclose(gf);
    }
    /* 4. stripes on an isometric grid */
    const double gs = ostep * pxs; int SW = (int)floor((stripe_px - 1) / gs) + 1; if (SW < 16) SW = 16;
    const int OW = (int)floor((umax - umin) / ostep) + 1, OH = (int)floor((vmax - vmin) / ostep) + 1;
    const int nstripe = OW <= SW ? 1 : (OW - 1 + SW - 2) / (SW - 1);
    snprintf(p, sizeof p, "%s/stripes.tsv", od); FILE *sf = fopen(p, "w");
    if (sf) fprintf(sf, "stripe\tout_col0\tout_cols\tout_rows\tu0_cm\tlength_cm\tvalid_cm2\tsrc_col0\tsrc_col1\twrap0\twrap1\n");
    long long nval_all = 0;
    for (int s = 0; s < nstripe; s++) {
        const double ts = snow();
        const int oc0 = s * (SW - 1), ow = oc0 + SW <= OW ? SW : OW - oc0; const double u0 = umin + oc0 * ostep, u1 = u0 + (ow - 1) * ostep;
        int la = F.lw, lb = -1;
        for (int j = 0; j < F.lw; j++) if (cumax[j] >= u0 - 4 * ostep && cumin[j] <= u1 + 4 * ostep) { if (j < la) la = j; if (j > lb) lb = j; }
        if (lb < la) continue;
        la = la > 0 ? la - 1 : 0; lb = lb + 2 < F.lw ? lb + 2 : F.lw;   /* [la, lb): the quads' left columns */
        const int c0 = la * F.D, c1 = ((lb - 1) * F.D + F.D < F.w - 1 ? (lb - 1) * F.D + F.D : F.w - 1), nc = c1 - c0 + 1;
        const size_t n = (size_t)nc * F.h; float *X = malloc(n * 4), *Y = malloc(n * 4), *Z = malloc(n * 4);
        if (!X || !Y || !Z || st_read(&F.S, c0, nc, nth, X, Y, Z)) { fprintf(stderr, "strip-flatten: read failed\n"); return 1; }
        const size_t on = (size_t)ow * OH; float *A[5];
        for (int k = 0; k < 5; k++) { A[k] = malloc(on * 4); if (!A[k]) { fprintf(stderr, "strip-flatten: out of memory\n"); return 1; } for (size_t i = 0; i < on; i++) A[k][i] = -1.f; }
        rsj R = {&F, X, Y, Z, c0, nc, la, lb, u0, vmin, ostep, ow, OH, A[0], A[1], A[2], A[3], A[4]};
        spfor((long)(lb - la) * (F.lh - 1), nth, k_rast, &R);
        free(X); free(Y); free(Z);
        long long nv = 0; double gmin = INFINITY, gmax = -INFINITY;
        for (size_t i = 0; i < on; i++) if (A[4][i] >= 0) { nv++; if (A[0][i] < gmin) gmin = A[0][i]; if (A[0][i] > gmax) gmax = A[0][i]; }
        nval_all += nv;
        if (flip) for (int k = 0; k < 5; k++) for (int y = 0; y < OH / 2; y++) { float *ra = A[k] + (size_t)y * ow, *rb = A[k] + (size_t)(OH - 1 - y) * ow;
            for (int x = 0; x < ow; x++) { const float t = ra[x]; ra[x] = rb[x]; rb[x] = t; } }
        char sn[64]; snprintf(sn, sizeof sn, "ws%03d", s); char dd[1500]; snprintf(dd, sizeof dd, "%s/stripes/%s", od, sn);
        if (tifxyz_write(dd, ow, OH, A[2], A[3], A[4], ostep, um, -1, "ufsm strip-flatten (whole-scroll strip: chunk SLIM blended, isometric uv grid)")) { fprintf(stderr, "strip-flatten: cannot write %s\n", dd); return 1; }
        snprintf(p, sizeof p, "%s/gu.tif", dd); if (tiffw_write(p, ow, OH, 3, A[0], 1)) return 1;
        snprintf(p, sizeof p, "%s/gv.tif", dd); if (tiffw_write(p, ow, OH, 3, A[1], 1)) return 1;
        const double vcm2 = nv * ostep * ostep * um * um * 1e-8;
        if (sf) { fprintf(sf, "%s\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.0f\t%.0f\t%d\t%d\n", sn, oc0, ow, OH, oc0 * ostep * um * 1e-4, (ow - 1) * ostep * um * 1e-4, vcm2, gmin, gmax,
                          nseg > 0 && nv ? seg_wrap(segs, nseg, gmin) : -1, nseg > 0 && nv ? seg_wrap(segs, nseg, gmax) : -1); fflush(sf); }
        fprintf(stderr, "strip-flatten: stripe %s %d x %d (u %.1f..%.1f cm), %.2f cm2 valid, strip columns %.0f..%.0f, %.1fs\n", sn, ow, OH, oc0 * ostep * um * 1e-4, (oc0 + ow - 1) * ostep * um * 1e-4, vcm2, gmin, gmax, snow() - ts);
        for (int k = 0; k < 5; k++) free(A[k]);
    }
    if (sf) fclose(sf);
    printf("strip-flatten: %d stripes of <= %d columns (%d px at render --scale %g), output grid %d x %d step %g, valid area %.2f cm2, %.1fs\n",
           nstripe, SW, render_out_px(SW, gs), pxs, OW, OH, ostep, nval_all * ostep * ostep * um * um * 1e-8, snow() - t0);
    snprintf(p, sizeof p, "%s/flatten.txt", od); FILE *ff = fopen(p, "w");
    if (ff) {
        fprintf(ff, "strip %d x %d step %g um %g decim %d stride %d\nlength_cm %.3f height_cm %.3f mesh_area_cm2 %.3f output %d x %d step %g valid_cm2 %.3f stripes %d flip_rows %d px_scale %g\n",
                F.w, F.h, F.step, um, F.D, S, L_cm, H_cm, A_cm2, OW, OH, ostep, nval_all * ostep * ostep * um * um * 1e-8, nstripe, flip, pxs);
        dstat_print(ff, "frame T", before); dstat_print(ff, "flattened", after); fclose(ff);
    }
    snprintf(p, sizeof p, "%s/flat.json", od); ff = fopen(p, "w");   /* (ufsm strip-mosaic reads it) */
    if (ff) { fprintf(ff, "{\"um\": %.9g, \"ostep\": %.9g, \"px_scale\": %.9g, \"flip_rows\": %d, \"stripes\": %d, \"length_cm\": %.4f, \"height_cm\": %.4f, \"decim\": %d, \"stride\": %d, "
                  "\"aniso_p50\": %.4f, \"aniso_p95\": %.4f, \"frame_aniso_p50\": %.4f, \"frame_aniso_p95\": %.4f, \"flips\": %lld}\n",
                  um, ostep, pxs, flip, nstripe, L_cm, H_cm, F.D, S, dstat_q(after, 0.5), dstat_q(after, 0.95), dstat_q(before, 0.5), dstat_q(before, 0.95), after->flips); fclose(ff); }
    free(before); free(after); free(sg); free(cumin); free(cumax); free(F.U); free(F.du); free(F.dv); free(F.wt); free(segs); st_close(&F.S);
    return 0;
}

/* ======================= ufsm strip-mosaic ======================= */
typedef struct { char name[64]; int ow, oh, gw, gh; double u0cm, gs; long long src0, src1; } ms_stripe;
static void hist_u8(const uint8_t *im, const uint8_t *m, size_t n, double *h) { for (size_t i = 0; i < n; i++) if (m[i]) h[im[i]]++; }
static void hist_q(const double *h, double lo, double hi, int *a, int *b) {
    double t = 0; for (int k = 0; k < 256; k++) t += h[k]; double c = 0; *a = 0; *b = 255;
    for (int k = 0; k < 256; k++) { c += h[k]; if (c < lo * t) *a = k + 1; if (c >= hi * t) { *b = k; break; } }
    if (*b <= *a) *b = *a + 1;
}
static inline uint8_t stretch(int v, int a, int b) { const int o = (v - a) * 255 / (b - a); return (uint8_t)(o < 0 ? 0 : o > 255 ? 255 : o); }
/* the stripe's valid mask at pixel resolution (a grid cell with 4 valid corners, as render) from the flat stripe's z grid */
static uint8_t *stripe_mask(const char *dir, int ow, int oh, double gs, int *gw, int *gh) {
    char p[1500]; snprintf(p, sizeof p, "%s/z.tif", dir); float *Z = img_read_f32(p, gw, gh); if (!Z) return nullptr;
    uint8_t *m = calloc((size_t)ow * oh, 1); const int w = *gw, h = *gh;
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
        int c = (int)(x / gs), r = (int)(y / gs); if (c > w - 2) c = w - 2; if (r > h - 2) r = h - 2;
        const size_t g = (size_t)r * w + c; m[(size_t)y * ow + x] = Z[g] >= 0 && Z[g + 1] >= 0 && Z[g + w] >= 0 && Z[g + w + 1] >= 0;
    }
    free(Z); return m;
}
/* strip-mosaic's per-stripe work on threads (stripes taken in turn; output and every sum identical to one thread) */
typedef struct {
    ms_stripe *st; int ns; const char *rd, *fd, *od; double gs; double *hc, *hi;
    int ca, cb, ia, ib, ctl, q, wstripes; double pxum; const st_seg *segs; int nseg;
    int rows; long long rowlen; int ds, ovw, bandh, rowh; double *ovs; float *ovn;
    int lds, lw, lh; double *lsum; float *lcnt; int tpx; long long ntile; double *tv, *tp; long long *tw0, *tw1;
    int wmin, nwr; double *wv, *wp, *vtot, *ptot;
    pthread_mutex_t mu; _Atomic int next, fail;
} ms_ctx;
static void *ms_pass1(void *arg) {
    ms_ctx *C = arg; char p[1600];
    for (int s; (s = atomic_fetch_add(&C->next, 1)) < C->ns;) {
        ms_stripe *S = &C->st[s];
        int w1, h1, w2, h2, gw, gh; snprintf(p, sizeof p, "%s/%s_ct.tif", C->rd, S->name); uint8_t *ct = img_read_u8(p, &w1, &h1);
        snprintf(p, sizeof p, "%s/%s.tif", C->rd, S->name); uint8_t *ink = img_read_u8(p, &w2, &h2);
        if (!ct || !ink || w1 != w2 || h1 != h2) { fprintf(stderr, "strip-mosaic: missing renders for %s\n", S->name); free(ct); free(ink); S->ow = 0; continue; }
        snprintf(p, sizeof p, "%s/stripes/%s", C->fd, S->name); uint8_t *m = stripe_mask(p, w1, h1, C->gs, &gw, &gh);
        double hc[256] = {0}, hi[256] = {0};
        if (m) { hist_u8(ct, m, (size_t)w1 * h1, hc); hist_u8(ink, m, (size_t)w1 * h1, hi); }
        pthread_mutex_lock(&C->mu); for (int k = 0; k < 256; k++) { C->hc[k] += hc[k]; C->hi[k] += hi[k]; } pthread_mutex_unlock(&C->mu);
        S->ow = w1; S->oh = h1; free(ct); free(ink); free(m);
    }
    return nullptr;
}
typedef struct { int w, h, ca, cb, ia, ib; const uint8_t *ct, *ink, *m; } ms_rows;
static void ms_stripe_row(void *u, int y, int x0, int n, uint8_t *dst) {   /* (called from the encoder's threads) */
    const ms_rows *R = u; const int k = y >= R->h; const size_t o = (size_t)(k ? y - R->h : y) * R->w + x0; const uint8_t *src = k ? R->ink : R->ct;
    for (int x = 0; x < n; x++) dst[x] = R->m[o + x] ? (k ? stretch(src[o + x], R->ia, R->ib) : stretch(src[o + x], R->ca, R->cb)) : 0;
}
static int ms_jpeg_stripe(const char *path, int w, int h, const uint8_t *ct, const uint8_t *ink, const uint8_t *m, int ca, int cb, int ia, int ib, int q) {
    /* CT over ink, stretched, 0 off the surface: rows generated on the fly by the banded multithreaded encoder (the same bytes as
       jpeg_write_gray of the stacked image, = libjpeg's) */
    ms_rows R = {.w = w, .h = h, .ca = ca, .cb = cb, .ia = ia, .ib = ib, .ct = ct, .ink = ink, .m = m};
    return jpar_write(path, w, 2 * h, 1, q, nullptr, ms_stripe_row, &R, 0);
}
static void *ms_pass2(void *arg) {
    ms_ctx *C = arg; char p[1600];
    for (int s; (s = atomic_fetch_add(&C->next, 1)) < C->ns;) {
        ms_stripe *S = &C->st[s]; if (!S->ow) continue;
        const double ts = snow(); const int w = S->ow, h = S->oh;
        int w1, h1, gw, gh; snprintf(p, sizeof p, "%s/%s_ct.tif", C->rd, S->name); uint8_t *ct = img_read_u8(p, &w1, &h1);
        snprintf(p, sizeof p, "%s/%s.tif", C->rd, S->name); uint8_t *ink = img_read_u8(p, &w1, &h1);
        snprintf(p, sizeof p, "%s/stripes/%s", C->fd, S->name); uint8_t *m = stripe_mask(p, w, h, C->gs, &gw, &gh);
        snprintf(p, sizeof p, "%s/stripes/%s/gu.tif", C->fd, S->name); int uw, uh; float *gu = img_read_f32(p, &uw, &uh);
        if (!ct || !ink || !m || !gu) { fprintf(stderr, "strip-mosaic: %s unreadable\n", S->name); atomic_store(&C->fail, 1); free(ct); free(ink); free(m); free(gu); continue; }
        const long long x0 = (long long)(S->u0cm * 1e4 / C->pxum + 0.5); const int xs = s ? 1 : 0;   /* (stripes share their first column with the previous one's last) */
        /* private ranges: overview cells by key row * ovw + ox (increasing with X), long-strip columns, tiles */
        const long long Xa = x0 + xs, Xb = x0 + w - 1;
        #define OVK(X) ((long long)((X) / C->rowlen) * C->ovw + (long long)(((X) - ((X) / C->rowlen) * C->rowlen) / C->ds))
        const long long ka = OVK(Xa), kb = OVK(Xb), nk = kb - ka + 1, la = Xa / C->lds, nl = Xb / C->lds - la + 1, ta = Xa / C->tpx, nt = Xb / C->tpx - ta + 1;
        const int bh2 = 2 * C->bandh;
        double *pos = calloc((size_t)nk * bh2, 8); float *pon = calloc((size_t)nk * bh2, 4);
        double *pls = calloc((size_t)nl * C->lh * 2, 8); float *plc = calloc((size_t)nl * C->lh, 4);
        double *ptv = calloc((size_t)nt, 8), *ptp = calloc((size_t)nt, 8); long long *pw0 = malloc((size_t)nt * 8), *pw1 = malloc((size_t)nt * 8);
        for (long long t = 0; t < nt; t++) { pw0[t] = 1 << 30; pw1[t] = -(1 << 30); }
        double *pwv = calloc((size_t)C->nwr + 1, 8), *pwp = calloc((size_t)C->nwr + 1, 8), vt = 0, pt = 0;
        for (int y = 0; y < h; y++) for (int x = xs; x < w; x++) {
            const size_t i = (size_t)y * w + x; if (!m[i]) continue;
            int mx = 0; for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { const int yy = y + dy, xx = x + dx; if (yy >= 0 && yy < h && xx >= 0 && xx < w && ct[(size_t)yy * w + xx] > mx) mx = ct[(size_t)yy * w + xx]; }
            const int pap = mx >= C->ctl; const long long X = x0 + x;
            const long long t = X / C->tpx - ta; ptv[t] += 1; ptp[t] += pap; vt += 1; pt += pap;
            int gc = (int)(x / C->gs), gr = (int)(y / C->gs); if (gc > uw - 1) gc = uw - 1; if (gr > uh - 1) gr = uh - 1;
            const float g = gu[(size_t)gr * uw + gc];
            if (g >= 0 && C->nseg > 0) { const int wr = seg_wrap(C->segs, C->nseg, g); if (wr < pw0[t]) pw0[t] = wr; if (wr > pw1[t]) pw1[t] = wr; pwv[wr - C->wmin] += 1; pwp[wr - C->wmin] += pap; }
            const int row = (int)(X / C->rowlen), ox = (int)((X - (long long)row * C->rowlen) / C->ds), oy = y / C->ds;
            if (row < C->rows && ox < C->ovw) { const size_t a = (size_t)(OVK(X) - ka) * bh2 + oy, b2 = a + C->bandh;
                                                pos[a] += stretch(ct[i], C->ca, C->cb); pon[a] += 1; pos[b2] += stretch(ink[i], C->ia, C->ib); pon[b2] += 1; }
            const long long lx = X / C->lds; const int ly = y / C->lds;
            if (lx < C->lw && ly < C->lh) { const size_t a = (size_t)(lx - la) * C->lh + ly; pls[2 * a] += stretch(ct[i], C->ca, C->cb); pls[2 * a + 1] += stretch(ink[i], C->ia, C->ib); plc[a] += 1; }
        }
        pthread_mutex_lock(&C->mu);
        for (long long k = 0; k < nk; k++) {
            const long long key = ka + k, row = key / C->ovw, ox = key % C->ovw; if (row >= C->rows) continue;
            for (int b = 0; b < bh2; b++) { const size_t a = (size_t)k * bh2 + b; if (!(pon[a] > 0)) continue;
                const size_t g = (size_t)(row * C->rowh + b) * C->ovw + ox; C->ovs[g] += pos[a]; C->ovn[g] += pon[a]; }
        }
        for (long long l = 0; l < nl; l++) { if (la + l >= C->lw) continue; for (int ly = 0; ly < C->lh; ly++) { const size_t a = (size_t)l * C->lh + ly; if (!(plc[a] > 0)) continue;
            const size_t g = (size_t)ly * C->lw + la + l; C->lsum[2 * g] += pls[2 * a]; C->lsum[2 * g + 1] += pls[2 * a + 1]; C->lcnt[g] += plc[a]; } }
        for (long long t = 0; t < nt; t++) { const long long g = ta + t; C->tv[g] += ptv[t]; C->tp[g] += ptp[t]; if (pw0[t] < C->tw0[g]) C->tw0[g] = pw0[t]; if (pw1[t] > C->tw1[g]) C->tw1[g] = pw1[t]; }
        for (int k = 0; k < C->nwr; k++) { C->wv[k] += pwv[k]; C->wp[k] += pwp[k]; }
        *C->vtot += vt; *C->ptot += pt;
        pthread_mutex_unlock(&C->mu);
        #undef OVK
        free(pos); free(pon); free(pls); free(plc); free(ptv); free(ptp); free(pw0); free(pw1); free(pwv); free(pwp);
        if (C->wstripes) { snprintf(p, sizeof p, "%s/%s_ct_ink.jpg", C->od, S->name); if (ms_jpeg_stripe(p, w, h, ct, ink, m, C->ca, C->cb, C->ia, C->ib, C->q)) fprintf(stderr, "strip-mosaic: cannot write %s\n", p); }
        fprintf(stderr, "strip-mosaic: %s %dx%d done, %.1fs\n", S->name, w, h, snow() - ts);
        free(ct); free(ink); free(m); free(gu);
    }
    return nullptr;
}
int cmd_strip_mosaic(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm strip-mosaic <flat-dir> <render-dir> <out-dir> [--strip DIR] [--ct-level 80] [--overview-px 16000] [--rows 10] [--tile-cm 1] [--quality 88] [--stripes 1] [--threads 8]\n"); return 2; }
    const char *fd = argv[2], *rd = argv[3], *od = argv[4];
    const int ctl = atoi(sopt(argc, argv, "--ct-level", "80")), ovpx = atoi(sopt(argc, argv, "--overview-px", "16000")), rows = atoi(sopt(argc, argv, "--rows", "10")), q = atoi(sopt(argc, argv, "--quality", "88"));
    const double tilecm = atof(sopt(argc, argv, "--tile-cm", "1")); const int wstripes = atoi(sopt(argc, argv, "--stripes", "1"));
    int nthr = atoi(sopt(argc, argv, "--threads", "8")); if (nthr < 1) nthr = 1; if (nthr > 64) nthr = 64;
    char p[1600], cmd[1600]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", od); if (system(cmd)) return 1;
    /* the strip's segments (wrap per source column): <flat-dir>/../strip.tsv or --strip */
    char sdir[1400]; snprintf(sdir, sizeof sdir, "%s", sopt(argc, argv, "--strip", "")); if (!*sdir) snprintf(sdir, sizeof sdir, "%s/..", fd);
    st_seg *segs = nullptr; const int nseg = st_segs(sdir, &segs);
    snprintf(p, sizeof p, "%s/stripes.tsv", fd); FILE *f = fopen(p, "r"); if (!f) { fprintf(stderr, "strip-mosaic: no %s\n", p); return 1; }
    snprintf(p, sizeof p, "%s/flat.json", fd); FILE *ft = fopen(p, "r"); double um = 0, ostep = 0, pxs = 1;
    if (ft) { char b[4096] = {0}; const size_t nr = fread(b, 1, sizeof b - 1, ft); (void)nr; fclose(ft); um = meta_num(b, "\"um\"", 0); ostep = meta_num(b, "\"ostep\"", 0); pxs = meta_num(b, "\"px_scale\"", 1); }
    if (!(um > 0) || !(ostep > 0)) { fprintf(stderr, "strip-mosaic: no %s (ufsm strip-flatten writes it)\n", p); return 1; }
    ms_stripe *st = calloc(4096, sizeof *st); int ns = 0; char line[2048];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    while (ns < 4096 && fgets(line, sizeof line, f)) {
        ms_stripe *s = &st[ns]; int oc0, oc, orr; double u0, len, vc; double g0, g1;
        if (sscanf(line, "%63s %d %d %d %lf %lf %lf %lf %lf", s->name, &oc0, &oc, &orr, &u0, &len, &vc, &g0, &g1) != 9) continue;
        s->gw = oc; s->gh = orr; s->u0cm = u0; s->src0 = (long long)g0; s->src1 = (long long)g1; ns++;
    }
    fclose(f);
    const double gs = ostep * pxs, pxum = um / pxs;   /* render pixels per grid step, um per pixel */
    fprintf(stderr, "strip-mosaic: %d stripes, %.2f um/px, CT level %d\n", ns, pxum, ctl);
    /* pass 1: global contrast (CT and ink over valid pixels); stripes on threads, each into its own histogram (integer counts: the sum
       is the same in any order) */
    double hc[256] = {0}, hi[256] = {0};
    ms_ctx C = {.st = st, .ns = ns, .rd = rd, .fd = fd, .od = od, .gs = gs, .hc = hc, .hi = hi};
    pthread_mutex_init(&C.mu, nullptr);
    { pthread_t th[64]; for (int t = 0; t < nthr; t++) pthread_create(&th[t], nullptr, ms_pass1, &C); for (int t = 0; t < nthr; t++) pthread_join(th[t], nullptr); }
    int ca, cb, ia, ib; hist_q(hc, 0.005, 0.998, &ca, &cb); hist_q(hi, 0.01, 0.999, &ia, &ib);
    fprintf(stderr, "strip-mosaic: stretch CT %d..%d, ink %d..%d\n", ca, cb, ia, ib);
    /* overview geometry: the whole length (pixels) folded into --rows rows of at most --overview-px */
    long long Ltot = 0; int Hpx = 0; for (int s = 0; s < ns; s++) if (st[s].ow) { Ltot = (long long)(st[s].u0cm * 1e4 / pxum + 0.5) + st[s].ow; if (st[s].oh > Hpx) Hpx = st[s].oh; }
    const long long rowlen = (Ltot + rows - 1) / rows; int ds = (int)((rowlen + ovpx - 1) / ovpx); if (ds < 1) ds = 1;
    while ((long long)rows * (2 * ((Hpx + ds - 1) / ds) + 16) > 65000) ds++;   /* (the JPEG limit on the folded height) */
    const int ovw = (int)((rowlen + ds - 1) / ds), bandh = (Hpx + ds - 1) / ds, barh = 6, rowh = 2 * bandh + barh + 10, ovh = rows * rowh;
    double *ovs = calloc((size_t)ovw * ovh, 8); float *ovn = calloc((size_t)ovw * ovh, 4);
    const int lds = (int)((Ltot + 65000 - 1) / 65000) < 1 ? 1 : (int)((Ltot + 65000 - 1) / 65000);   /* the one-row strip */
    const int lw = (int)((Ltot + lds - 1) / lds), lh = (Hpx + lds - 1) / lds; double *lsum = calloc((size_t)lw * lh * 2, 8); float *lcnt = calloc((size_t)lw * lh, 4);
    /* papyrus along the strip: per tile-cm column of the strip (all rows) and per wrap */
    const int tpx = (int)lround(tilecm * 1e4 / pxum); const long long ntile = Ltot / tpx + 1;
    double *tv = calloc((size_t)ntile, 8), *tp = calloc((size_t)ntile, 8); long long *tw0 = malloc((size_t)ntile * 8), *tw1 = malloc((size_t)ntile * 8);
    for (long long t = 0; t < ntile; t++) { tw0[t] = 1 << 30; tw1[t] = -(1 << 30); }
    int wmin = 1 << 30, wmax = -(1 << 30); for (int k = 0; k < nseg; k++) { if (segs[k].wrap < wmin) wmin = segs[k].wrap; if (segs[k].wrap > wmax) wmax = segs[k].wrap; }
    const int nwr = nseg > 0 ? wmax - wmin + 1 : 0; double *wv = calloc((size_t)nwr + 1, 8), *wp = calloc((size_t)nwr + 1, 8);
    double vtot = 0, ptot = 0;
    /* pass 2: stripes on threads; each accumulates into private arrays over its own range of overview / long-strip cells, tiles and
       wraps, merged under a lock (sums of integers: identical to the serial order) */
    C.ca = ca; C.cb = cb; C.ia = ia; C.ib = ib; C.ctl = ctl; C.q = q; C.wstripes = wstripes; C.pxum = pxum; C.segs = segs; C.nseg = nseg;
    C.rows = rows; C.rowlen = rowlen; C.ds = ds; C.ovw = ovw; C.bandh = bandh; C.rowh = rowh; C.ovs = ovs; C.ovn = ovn;
    C.lds = lds; C.lw = lw; C.lh = lh; C.lsum = lsum; C.lcnt = lcnt; C.tpx = tpx; C.ntile = ntile; C.tv = tv; C.tp = tp; C.tw0 = tw0; C.tw1 = tw1;
    C.wmin = wmin; C.nwr = nwr; C.wv = wv; C.wp = wp; C.vtot = &vtot; C.ptot = &ptot; atomic_store(&C.next, 0);
    { pthread_t th[64]; for (int t = 0; t < nthr; t++) pthread_create(&th[t], nullptr, ms_pass2, &C); for (int t = 0; t < nthr; t++) pthread_join(th[t], nullptr); }
    if (atomic_load(&C.fail)) return 1;
    /* the overview: per row CT band, ink band, papyrus bar (green on papyrus .. red off, grey no surface) */
    uint8_t *ov = malloc((size_t)ovw * ovh * 3); memset(ov, 24, (size_t)ovw * ovh * 3);
    for (size_t i = 0; i < (size_t)ovw * ovh; i++) if (ovn[i] > 0) { const uint8_t v = (uint8_t)lround(ovs[i] / ovn[i]); ov[3 * i] = ov[3 * i + 1] = ov[3 * i + 2] = v; }
    for (int r = 0; r < rows; r++) for (int x = 0; x < ovw; x++) {
        const long long X0 = (long long)r * rowlen + (long long)x * ds, t = X0 / tpx; if (X0 >= Ltot || t >= ntile) continue;
        uint8_t col[3] = {90, 90, 90};
        if (tv[t] > 0) { const double fr = tp[t] / tv[t]; col[0] = (uint8_t)lround(255 * fmin(1, 2 * (1 - fr))); col[1] = (uint8_t)lround(255 * fmin(1, 2 * fr)); col[2] = 0; }
        for (int y = 0; y < barh; y++) { const size_t i = (size_t)(r * rowh + 2 * bandh + 2 + y) * ovw + x; memcpy(ov + 3 * i, col, 3); }
        if (t * tpx >= X0 && t * tpx < X0 + ds && t % 10 == 0) for (int y = 0; y < 3; y++) { const size_t i = (size_t)(r * rowh + 2 * bandh + 2 + barh + y) * ovw + x; ov[3 * i] = ov[3 * i + 1] = ov[3 * i + 2] = 230; }   /* 10 cm ticks */
    }
    snprintf(p, sizeof p, "%s/overview.jpg", od); if (jpeg_write_rgb(p, ovw, ovh, ov, q)) fprintf(stderr, "strip-mosaic: cannot write %s\n", p);
    free(ov);
    uint8_t *ln = malloc((size_t)lw * lh * 2 * 3);
    for (int y = 0; y < lh; y++) for (int x = 0; x < lw; x++) { const size_t a = (size_t)y * lw + x; const uint8_t c = lcnt[a] > 0 ? (uint8_t)lround(lsum[2 * a] / lcnt[a]) : 24, k = lcnt[a] > 0 ? (uint8_t)lround(lsum[2 * a + 1] / lcnt[a]) : 24;
        const size_t i0 = (size_t)y * lw + x, i1 = (size_t)(lh + y) * lw + x; ln[3 * i0] = ln[3 * i0 + 1] = ln[3 * i0 + 2] = c; ln[3 * i1] = ln[3 * i1 + 1] = ln[3 * i1 + 2] = k; }
    snprintf(p, sizeof p, "%s/strip_long.jpg", od); if (jpeg_write_rgb(p, lw, 2 * lh, ln, q)) fprintf(stderr, "strip-mosaic: cannot write %s\n", p);
    free(ln);
    snprintf(p, sizeof p, "%s/strip_papyrus.tsv", od); FILE *pf = fopen(p, "w");
    if (pf) { fprintf(pf, "u_cm\tvalid_cm2\ton_papyrus\twrap_lo\twrap_hi\n");
              for (long long t = 0; t < ntile; t++) if (tv[t] > 0) fprintf(pf, "%.2f\t%.4f\t%.4f\t%lld\t%lld\n", t * tilecm, tv[t] * pxum * pxum * 1e-8, tp[t] / tv[t], tw0[t], tw1[t]);
              fclose(pf); }
    snprintf(p, sizeof p, "%s/papyrus_wraps.tsv", od); pf = fopen(p, "w");
    if (pf) { fprintf(pf, "wrap\tvalid_cm2\ton_papyrus\n"); for (int k = 0; k < nwr; k++) if (wv[k] > 0) fprintf(pf, "w%03d\t%.3f\t%.4f\n", wmin + k, wv[k] * pxum * pxum * 1e-8, wp[k] / wv[k]); fclose(pf); }
    printf("strip-mosaic: %d stripes, length %.2f cm, valid %.2f cm2, on papyrus (CT >= %d within 3x3 px) %.4f; overview %s/overview.jpg %dx%d (%d rows, 1:%d), long strip %dx%d (1:%d)\n",
           ns, Ltot * pxum * 1e-4, vtot * pxum * pxum * 1e-8, ctl, vtot > 0 ? ptot / vtot : 0, od, ovw, ovh, rows, ds, lw, 2 * lh, lds);
    free(st); free(ovs); free(ovn); free(lsum); free(lcnt); free(tv); free(tp); free(tw0); free(tw1); free(wv); free(wp); free(segs);
    return 0;
}

/* ufsm strip-crop <in.tif> <out.jpg> x,y,w,h [--stretch 1]: a rectangle of a big uint8 render TIFF (stripes are 60000 px wide: past the
   usual image tools' limits) as a grayscale JPEG, percentile-stretched over its nonzero pixels */
int cmd_strip_crop(int argc, char **argv) {
    int c[4];
    if (argc < 5 || sscanf(argv[4], "%d,%d,%d,%d", c, c + 1, c + 2, c + 3) != 4) { fprintf(stderr, "usage: ufsm strip-crop <in.tif> <out.jpg> x,y,w,h [--stretch 1]\n"); return 2; }
    int w, h; uint8_t *im = img_read_u8(argv[2], &w, &h); if (!im) { fprintf(stderr, "strip-crop: cannot read %s\n", argv[2]); return 1; }
    if (c[0] < 0) c[0] = 0; if (c[1] < 0) c[1] = 0; if (c[0] + c[2] > w) c[2] = w - c[0]; if (c[1] + c[3] > h) c[3] = h - c[1];
    if (c[2] < 1 || c[3] < 1) { fprintf(stderr, "strip-crop: empty rectangle\n"); free(im); return 2; }
    uint8_t *o = malloc((size_t)c[2] * c[3]), *m = malloc((size_t)c[2] * c[3]);
    for (int y = 0; y < c[3]; y++) for (int x = 0; x < c[2]; x++) { const uint8_t v = im[(size_t)(c[1] + y) * w + c[0] + x]; o[(size_t)y * c[2] + x] = v; m[(size_t)y * c[2] + x] = v > 0; }
    if (atoi(sopt(argc, argv, "--stretch", "1"))) img_stretch(o, m, (size_t)c[2] * c[3], 0.01, 0.995);
    const int rc = jpeg_write_gray(argv[3], c[2], c[3], o, 90);
    free(im); free(o); free(m); return rc ? 1 : 0;
}
