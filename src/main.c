/* ufsm CLI. */
#include "sample.h"
#include "sources.h"
#include "band.h"
#include "votes.h"
#include "wind.h"
#include "edt.h"
#include "store.h"
#include "zarr3.h"
#include "fit.h"
#include "fblend.h"
#include "json.h"
#include "nn.h"
#include "tiffw.h"
#include "z3w.h"
#include "papyrus.h"
#include <zlib.h>
#include "tiff.h"
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int cmd_ingest_labels(int argc, char **argv);
int cmd_ingest_kaggle(int argc, char **argv);
int cmd_ingest_mesh(int argc, char **argv);
int cmd_ls(int argc, char **argv);
int cmd_ingest_zip(int argc, char **argv);
int cmd_raster(int argc, char **argv);
int cmd_train(int argc, char **argv);
int cmd_predict(int argc, char **argv);
int cmd_eval(int argc, char **argv);
int cmd_refine(int argc, char **argv);       /* src/surf.c */
int cmd_surf_stats(int argc, char **argv);
int cmd_stitch(int argc, char **argv);   /* src/stitch.c */
int cmd_coreline(int argc, char **argv);   /* src/coreline.c */

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int arg_index(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }

static int usage(void) {
    fprintf(stderr,
        "ufsm — ultra fast scroll model\n"
        "  ufsm info    <root> <array-or-group-key> [--cache DIR]\n"
        "  ufsm read    <root> <array-key> z y x nz ny nx <out.raw> [--cache DIR] [--threads N]\n"
        "  ufsm slice   <root> <array-key> z <out.pgm> [--cache DIR] [--threads N]     (whole z slice)\n"
        "  ufsm regions <root> <regions-dir-key>                 -> JSON list of region origins\n"
        "  ufsm axis    <root> <ct-group-key> <out.json> [--cache DIR]  -> per-z centroid axis (umbilicus stand-in)\n"
        "  ufsm ingest-labels <root> <zarr-key> <out-dir> --um U   HF/S3 zarr v2 labels -> volcomp pyramid\n"
        "  ufsm ingest-zip <labels.zip> <zarr-name> <out-dir> --um U  label zarr from the HF archive -> volcomp pyramid\n"
        "  ufsm raster <out-dir> --shape Z,Y,X --um U [--level L] [--T 3] <mesh.sfc|tifxyz>...  meshes -> label pyramid\n"
        "  ufsm ingest-kaggle <hf-root> <out-dir> [--n N]          Kaggle cubes -> volcomp images + labels\n"
        "  ufsm ingest-mesh <root> <tifxyz-key> <out.sfc>           tifxyz -> surfcomp\n"
        "  ufsm train  <sources.json> --out DIR [--P 512 --B 1 --steps N --gpus 0 ...]   train the recto model\n"
        "  ufsm predict <ckpt> <root> <ct-group> <out-dir> --um U [--box ...]      sliding-window inference -> volcomp pyramid\n"
        "  ufsm eval <pred-root> <pred-group> <label-root> <label-group> --um U [--box ...]   precision/recall/dice\n"
        "  ufsm sample  <sources.json> [--P 128] [--B 2] [--n 4] [--seed S] [--workers W] [--out DIR] [--noaug]\n"
        "  ufsm prefetch <sources.json> [--levels 1] [--threads 32] [--fraction 1]   warm the CT chunk cache for the labelled cells\n"
        "  ufsm phase-raster <codes-root> <key> <out-dir> [--box ...]          band phase precomputed for the sampler\n"
        "  ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase|votes  one array halved per axis (labels, phase, side votes)\n"
        "  ufsm side-votes <out-group> <out-um> <root> <key> [<root> <key> ...]   several surface sets' side labels added into one vote array\n"
        "  ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>...   other authors' surfaces get winding q in a reference raster's gauge\n"
        "  ufsm codes-merge <out-group> <out-um> <root> <key> ...   winding code rasters united (same wrap within 1/4 turn, else 255)\n"
        "  ufsm ct-hist <out.txt> <root> <key> z,y,x,n [...]  CT histograms of boxes (targets of train --aug-hist, the hist op)\n"
        "  ufsm fit <fields-dir> [--slab Z] [--out MODEL] [--winding ZARR] [--codes ROOT KEY]   winding fitter (src/fit.h)\n"
        "  ufsm wraps <model-dir> <out-dir> [--step 8] [--fields DIR]          one tifxyz per wrap (level sets of W)\n"
        "  ufsm coreline detect|box|sdist ...   a crushed scroll's flattened umbilicus per z (fit fields) and half-box fit dirs behind it\n"
        "  ufsm stitch align|solve|field|sheets|info ...   overlapping box fits -> one gauge, a global winding field, stitched sheets\n"
        "  ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY]  ink / CT along each wrap (tiled TIFF)\n"
        "  ufsm wrapdist <wraps-dir> <labels-root> <labels-key>              wrap vertices vs the nearest labelled sheet\n"
        "  ufsm wrapswitch <wraps-dir> <codes-root> <codes-key>            how often the wraps switch sheets (per cm)\n"
        "  ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U   tifxyz vertices onto the nearest papyrus ridge\n"
        "  ufsm refine <wraps-in> <wraps-out> --fields DIR [--range 0.33] [--smooth-um 40]   vertices onto the fields' sheet phase\n"
        "  ufsm surf-stats <wraps-dir> --fields DIR [--ct ROOT KEY --ct-um U]   label-free surface quality (phase, switches, CT)\n"
        "  ufsm fit-compare <reference-fields> <fields>                      fit fields vs a reference (phase, spacing, normals)\n"
        "  ufsm field-blend <out> <fields1> <fields2> ... [--weights a,b,..]  several models' fit fields averaged (phase: confidence-weighted complex mean)\n"
        "  ufsm wind-score <sweep-dir> <codes-root> <codes-key>             the fit --codes score of a predict --cond-sweep winding raster\n"
        "  ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx   papyrus / void training labels of a CT box (gap labels, no cutoff)\n"
        "  ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w --labels G1,G2   a CT plane beside label panels (void blue, ignored yellow)\n"
        "root: local dir or https://.../volcomp ; keys are paths under root\n");
    return 2;
}

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return 1;
    return 0;
}

static int write_pgm(const char *path, const uint8_t *img, int64_t w, int64_t h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%lld %lld\n255\n", (long long)w, (long long)h);
    fwrite(img, 1, (size_t)(w * h), f);
    fclose(f);
    return 0;
}

/* ---- regions: list region_<z>_<y>_<x>.zarr under a directory (local readdir or nginx index html) ---- */
static int cmd_regions(store *s, const char *key) {
    int n = 0;
    printf("[");
    if (store_is_local(s)) {
        char p[1200];
        snprintf(p, sizeof p, "%s/%s", store_root(s), key);
        DIR *d = opendir(p);
        if (!d) { fprintf(stderr, "cannot open %s\n", p); return 1; }
        struct dirent *e;
        while ((e = readdir(d))) {
            long long z, y, x;
            if (sscanf(e->d_name, "region_%lld_%lld_%lld.zarr", &z, &y, &x) == 3 && !strstr(e->d_name, ".part"))
                printf("%s[%lld,%lld,%lld]", n++ ? ",\n " : "", z, y, x);
        }
        closedir(d);
    } else {
        size_t len;
        char k[1200];
        snprintf(k, sizeof k, "%s/", key);
        uint8_t *html = store_read_all(s, k, &len);
        if (!html) { fprintf(stderr, "cannot list %s\n", key); return 1; }
        for (char *p = (char *)html; (p = strstr(p, "href=\"region_")); p += 6) {
            long long z, y, x;
            char tail[16] = "";
            if (sscanf(p + 6, "region_%lld_%lld_%lld.zarr%15[^\"]", &z, &y, &x, tail) >= 3 && !strcmp(tail, "/"))
                printf("%s[%lld,%lld,%lld]", n++ ? ",\n " : "", z, y, x);
        }
        free(html);
    }
    printf("]\n");
    fprintf(stderr, "%d regions\n", n);
    return 0;
}

/* ---- axis: per-z centroid of the masked CT at a coarse level, written as umbilicus control points ---- */
static int cmd_axis(store *s, const char *key, const char *out, const char *cache) {
    z3_level lv[16];
    int nl = z3_group_levels(s, key, lv, 16);
    z3 *z = nullptr;
    int level = -1;
    for (int l = 0; l < (nl > 0 ? nl : MAXLEV); l++) {   /* coarsest level whose slices are still >= 128 wide */
        char ak[1200];
        snprintf(ak, sizeof ak, "%s/%d", key, l);
        z3 *t = z3_open(s, ak, cache);
        if (!t) break;
        if (z && z3_meta_of(t)->shape[1] < 128) { z3_close(t); break; }
        z3_close(z);
        z = t;
        level = l;
    }
    if (!z) { fprintf(stderr, "axis: no readable level under %s\n", key); return 1; }
    const z3_meta *m = z3_meta_of(z);
    int64_t Z = m->shape[0], Y = m->shape[1], X = m->shape[2];
    fprintf(stderr, "axis: level %d, %lld x %lld x %lld\n", level, (long long)Z, (long long)Y, (long long)X);
    uint8_t *slab = malloc((size_t)m->shard[0] * Y * X);
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "{\"control_points\":[");
    int n = 0;
    double scale = (double)(1 << level);
    for (int64_t z0 = 0; z0 < Z; z0 += m->shard[0]) {
        int64_t nz = z0 + m->shard[0] > Z ? Z - z0 : m->shard[0];
        int64_t o[3] = {z0, 0, 0}, cnt[3] = {nz, Y, X};
        if (z3_read(z, o, cnt, slab, 0)) { fprintf(stderr, "axis: %s\n", z3_error()); return 1; }
        for (int64_t zi = 0; zi < nz; zi += 4) {
            const uint8_t *sl = slab + (size_t)zi * Y * X;
            double sy = 0, sx = 0; int64_t k = 0;
            for (int64_t y = 0; y < Y; y++)
                for (int64_t x = 0; x < X; x++)
                    if (sl[y * X + x]) { sy += (double)y; sx += (double)x; k++; }
            if (k < Y * X / 100) continue;
            fprintf(f, "%s{\"z\":%.1f,\"y\":%.1f,\"x\":%.1f,\"score\":%lld}", n++ ? "," : "", (double)(z0 + zi) * scale,
                    sy / (double)k * scale, sx / (double)k * scale, (long long)k);
        }
    }
    fprintf(f, "],\"method\":\"ufsm axis: per-z centroid of CT>0 at level %d\"}\n", level);
    fclose(f);
    free(slab);
    z3_close(z);
    fprintf(stderr, "axis: %d control points -> %s\n", n, out);
    return 0;
}

/* ---- sample: dump montages of drawn patches ---- */
/* the winding fitter's input fields (src/fit.h) from the label-grid phase and spacing of a native box b (oracle fields): per field
   voxel of h native voxels (h / 2 label voxels per axis) the mean unit phasor of the known phase (pc, ps: 128 + 127 * mean),
   the mean n n^T of the wrap-aware phase gradient (central differences over one label voxel), the mean spacing 1 / |grad phase|
   (units of h / 8: 2 * label spacing * 8 / h) and the known fraction (w); fields.json records the box, h and the axis table. */
static int band_fit_fields(const uint8_t *ph, const uint8_t *spc, const int n[3], int halo, const long long b[6], int h, const axis *ax, const char *dir) {
    const int bl = h / 2, D = (int)(b[3] / h), H = (int)(b[4] / h), W = (int)(b[5] / h);
    const size_t nf = (size_t)D * H * W, pz = (size_t)n[1] * n[2];
    uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(nf, 1);
    for (int fz = 0; fz < D; fz++) for (int fy = 0; fy < H; fy++) for (int fx = 0; fx < W; fx++) {
        double sc = 0, ss = 0, t[6] = {0}, sp = 0; int kn = 0, kt = 0, ksp = 0;
        for (int dz = 0; dz < bl; dz++) for (int dy = 0; dy < bl; dy++) for (int dx = 0; dx < bl; dx++) {
            const int z = halo + fz * bl + dz, y = halo + fy * bl + dy, x = halo + fx * bl + dx;
            const size_t i = (size_t)z * pz + (size_t)y * n[2] + x;
            if (ph[i] >= PHASE_PERIOD) continue;
            const double q = 2 * M_PI * ph[i] / PHASE_PERIOD; sc += cos(q); ss += sin(q); kn++;
            const uint8_t nb[6] = {ph[i + pz], ph[i - pz], ph[i + n[2]], ph[i - n[2]], ph[i + 1], ph[i - 1]};
            int ok = 1; for (int k = 0; k < 6; k++) ok &= nb[k] < PHASE_PERIOD;
            if (!ok) continue;
            double g[3];
            for (int d = 0; d < 3; d++) { int df = (int)nb[2 * d] - (int)nb[2 * d + 1]; df -= PHASE_PERIOD * (df > PHASE_PERIOD / 2); df += PHASE_PERIOD * (df <= -PHASE_PERIOD / 2); g[d] = df; }
            const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (gn < 1e-9) continue;
            sp += 2.0 * PHASE_PERIOD / gn; ksp++;   /* spacing (label voxels per turn) from the gradient: 2 * 252 / |central difference| */
            for (int d = 0; d < 3; d++) g[d] /= gn;
            t[0] += g[0] * g[0]; t[1] += g[1] * g[1]; t[2] += g[2] * g[2]; t[3] += g[0] * g[1]; t[4] += g[0] * g[2]; t[5] += g[1] * g[2]; kt++;
        }
        const size_t fi = ((size_t)fz * H + fy) * W + fx;
        if (!kn) { a[0][fi] = a[1][fi] = 128; continue; }
        a[0][fi] = (uint8_t)lround(128 + 127 * sc / kn); a[1][fi] = (uint8_t)lround(128 + 127 * ss / kn);
        if (kt) for (int k = 0; k < 6; k++) a[2 + k][fi] = (uint8_t)lround(255 * fmin(1, fmax(0, (k < 3 ? 0 : 0.5) + t[k] / kt)));
        if (ksp) { const double v = 2.0 * sp / ksp * 8 / h; a[8][fi] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); }
        a[9][fi] = (uint8_t)lround(255.0 * kn / ((double)bl * bl * bl));
    }
    const int na = D * h + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { double y, x; axis_at(ax, (double)(b[0] + z), &y, &x); cy[z] = (float)(y - b[1]); cx[z] = (float)(x - b[2]); }
    const long long org[3] = {b[0], b[1], b[2]}; const int shp[3] = {D, H, W};
    const int rc = fit_fields_save(dir, org, shp, h, 2.4, a, cy, cx, na);   /* the band field: 2.4 um native (4.8 um labels) */
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx);
    return rc;
}

/* ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256] [--shards 1]
   [--pair 1|2]
   --pair 2: recto / verso side labels instead (band_field_side_pair: 0 unknown, 1 verso, 2 recto; between consecutive wraps k and
   k + 1 the half nearer k recto; halo 2 radius), fill 0, for train sources with "encoding": "side".
   The band field's winding phase (src/band.h PHASE_*, what the sampler computes per window for --side 2 / task unwrap) of a whole
   winding_mod14 raster, on its label grid: written as <out-dir>/4.8 (uint8, lossless volcomp, fill PHASE_UNKNOWN) plus the group;
   a source then reads it with "band": {..., "phase": {"root": <out-dir>, "key": "4.8"}}. Blocks of --block label voxels plus the
   band halo, in parallel, shard by shard (512 label voxels); blocks without codes stay unknown. --box (native) limits it.
   --pool 2 writes only the 2x circular-mean pooling, <out-dir>/9.6 (a whole scroll at 4.8 um is ~25 MB per shard, 9.6 um
   ~5 MB); --resume 1 skips shards already written (long runs). --winding 1 writes the winding mod 14 turns (band_winding, same
   encoding and pooling: 1/18 turn steps) instead of the 1-turn phase, for targets beyond one turn. --gs 4 computes the field on the
   codes pooled 2x (praster_read: circular mean, conflicts kept) into <out-dir>/9.6: 35x faster (block B 7.5 s vs 4.4 min); against the
   4.8 um phase pooled to 9.6 on B: mean 0.016 turn apart, 1% over 0.1 turn. A source reads it for 2.4 um windows with "phase":
   {"key": "9.6", "level": 2} (sample.c phase_window_interp: circular trilinear 4x). */
typedef struct { z3 *codes; const int64_t *shape; const axis *ax; band_params bp; int halo, blk; int64_t s0[3]; uint8_t *shard; atomic_int next; int nblk[3]; size_t nonempty; pthread_mutex_t mu; int winding; int pf; const int64_t *fshape; int pair; } praster_job;   /* pf 2: the grid is the 4.8 um codes (fshape) pooled 2x */
/* the codes over the grid box ro / rn: read directly (pf 1), or from the 2x finer codes with every 2^3 block pooled (pf 2): empty when
   all are empty, 255 (conflict) when any is or two lie more than a quarter turn apart, else their circular mean */
static int praster_read(const praster_job *J, const int64_t ro[3], const int64_t rn[3], uint8_t *out) {
    if (J->pf == 1) return z3_read(J->codes, ro, rn, out, 1);
    static float cs[BAND_PERIOD], sn[BAND_PERIOD]; static _Atomic int init;
    if (!atomic_load(&init)) { for (int k = 0; k < BAND_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / BAND_PERIOD); sn[k] = (float)sin(2 * M_PI * k / BAND_PERIOD); } atomic_store(&init, 1); }
    int64_t fo[3], fn[3];
    for (int d = 0; d < 3; d++) { fo[d] = 2 * ro[d]; const int64_t e = 2 * (ro[d] + rn[d]) < J->fshape[d] ? 2 * (ro[d] + rn[d]) : J->fshape[d]; fn[d] = e - fo[d]; }
    memset(out, 0, (size_t)rn[0] * rn[1] * rn[2]);
    if (fn[0] <= 0 || fn[1] <= 0 || fn[2] <= 0) return 0;
    uint8_t *f = malloc((size_t)fn[0] * fn[1] * fn[2]);
    if (!f || z3_read(J->codes, fo, fn, f, 1)) { free(f); return -1; }
    for (int64_t a = 0; a < rn[0]; a++) for (int64_t b = 0; b < rn[1]; b++) for (int64_t c = 0; c < rn[2]; c++) {
        int first = -1, conflict = 0, k = 0; float x = 0, y = 0;
        for (int q = 0; q < 8 && !conflict; q++) {
            const int64_t z = 2 * a + (q >> 2 & 1), yy = 2 * b + (q >> 1 & 1), xx = 2 * c + (q & 1);
            if (z >= fn[0] || yy >= fn[1] || xx >= fn[2]) continue;
            const int v = f[((size_t)z * fn[1] + yy) * fn[2] + xx];
            if (!v) continue;
            if (v == 255 || (first >= 0 && 2 * abs(band_diff(v - 1, first)) > BAND_STEPS / 2)) { conflict = 1; break; }
            if (first < 0) first = v - 1;
            x += cs[v - 1]; y += sn[v - 1]; k++;
        }
        uint8_t o = 0;
        if (conflict) o = 255;
        else if (k) { double t = atan2(y, x) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)(1 + (int)lround(t * BAND_PERIOD) % BAND_PERIOD); }
        out[((size_t)a * rn[1] + b) * rn[2] + c] = o;
    }
    free(f);
    return 0;
}
static void *praster_worker(void *arg) {
    praster_job *J = arg; const int B = J->blk, Hh = J->halo, nb = J->nblk[0] * J->nblk[1] * J->nblk[2];
    for (;;) {
        const int t = atomic_fetch_add(&J->next, 1); if (t >= nb) break;
        const int bz = t / (J->nblk[1] * J->nblk[2]), by = (t / J->nblk[2]) % J->nblk[1], bx = t % J->nblk[2];
        int64_t o[3], n[3], ro[3], rn[3]; int ni[3]; int ok = 1;
        const int64_t b0[3] = {J->s0[0] + (int64_t)bz * B, J->s0[1] + (int64_t)by * B, J->s0[2] + (int64_t)bx * B};
        for (int d = 0; d < 3; d++) {
            if (b0[d] >= J->shape[d]) ok = 0;
            o[d] = b0[d] - Hh; n[d] = B + 2 * Hh; ni[d] = (int)n[d];
            ro[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + n[d] < J->shape[d] ? o[d] + n[d] : J->shape[d]; rn[d] = e - ro[d];
            if (rn[d] <= 0) ok = 0;
        }
        if (!ok) continue;
        const size_t N = (size_t)n[0] * n[1] * n[2];
        uint8_t *codes = calloc(N, 1), *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (!codes || !tmp || praster_read(J, ro, rn, tmp)) { free(codes); free(tmp); fprintf(stderr, "phase-raster: read failed: %s\n", z3_error()); continue; }
        size_t nz = 0; for (size_t k = 0; k < (size_t)rn[0] * rn[1] * rn[2]; k++) nz += tmp[k] != 0;
        if (!nz) { free(codes); free(tmp); continue; }
        for (int64_t zz = 0; zz < rn[0]; zz++) for (int64_t yy = 0; yy < rn[1]; yy++)
            memcpy(codes + ((size_t)(zz + ro[0] - o[0]) * n[1] + (yy + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), tmp + ((size_t)zz * rn[1] + yy) * rn[2], (size_t)rn[2]);
        free(tmp);
        double *cy = malloc(n[0] * sizeof(double)), *cx = malloc(n[0] * sizeof(double));
        const double g = band_gs(J->bp); for (int64_t zz = 0; zz < n[0]; zz++) axis_at(J->ax, g * (zz + o[0]) + 0.5 * (g - 1), &cy[zz], &cx[zz]);
        uint8_t *band = malloc(N), *phase = malloc(N);
        if (!band || !phase || (J->pair == 2 ? band_field_side_pair(codes, ni, J->bp.radius / band_gs(J->bp), Hh, phase) : J->pair ? band_field_phase_pair(codes, ni, J->bp.radius / band_gs(J->bp), phase, nullptr) : band_field_phase(codes, ni, o, cy, cx, J->bp, band, phase, nullptr))) { fprintf(stderr, "phase-raster: out of memory\n"); free(codes); free(band); free(phase); free(cy); free(cx); continue; }
        if (J->winding) for (size_t k = 0; k < N; k++) phase[k] = band_winding(band[k], phase[k]);   /* --winding: mod 14 turns */
        for (int zz = 0; zz < B; zz++) for (int yy = 0; yy < B; yy++)   /* the block interior into the shard buffer (512^3) */
            memcpy(J->shard + ((size_t)(bz * B + zz) * 512 + by * B + yy) * 512 + bx * B, phase + ((size_t)(zz + Hh) * n[1] + yy + Hh) * n[2] + Hh, (size_t)B);
        pthread_mutex_lock(&J->mu); J->nonempty++; pthread_mutex_unlock(&J->mu);
        free(codes); free(band); free(phase); free(cy); free(cx);
    }
    return nullptr;
}
/* the 2x-pooled band phase of a 512^3 label-grid shard into 256^3: per 2^3 block the circular mean of the known codes,
   PHASE_UNKNOWN where none is known or their resultant is under half their count (as ufsm downsample --mode phase) */
static void phase_pool2(const uint8_t *src, uint8_t *dst) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static int init;
    if (!init) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } init = 1; }
    for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) {
        float c = 0, sv = 0; int known = 0;
        for (int k = 0; k < 8; k++) {
            const int v = src[((size_t)(2 * z + (k >> 2 & 1)) * 512 + 2 * y + (k >> 1 & 1)) * 512 + 2 * x + (k & 1)];
            if (v < PHASE_PERIOD) { c += cs[v]; sv += sn[v]; known++; }
        }
        uint8_t o = PHASE_UNKNOWN;
        if (known && c * c + sv * sv >= 0.25f * known * known) { double t = atan2(sv, c) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)((int)lround(t * PHASE_PERIOD) % PHASE_PERIOD); }
        dst[((size_t)z * 256 + y) * 256 + x] = o;
    }
}
/* phase-raster: shards taken from list in turn, each by nt block threads (praster_worker) into its own buffer, then (pooled) written */
typedef struct { z3 *z; const int64_t *shape; const axis *ax; band_params bp; int halo, blk, winding, pf, pair; const int64_t *fshape; z3w *w; int pool, nt;
                 const int64_t (*list)[3]; size_t nlist; atomic_size_t next; size_t nonempty, done; double t0; pthread_mutex_t mu; atomic_int fail; } praster_run;
static void *praster_shards(void *arg) {
    praster_run *R = arg;
    uint8_t *shard = malloc((size_t)512 * 512 * 512), *pooled = R->pool == 2 ? malloc((size_t)256 * 256 * 256) : nullptr;
    if (!shard || (R->pool == 2 && !pooled)) { atomic_store(&R->fail, 1); free(shard); free(pooled); return nullptr; }
    for (;;) {
        const size_t t = atomic_fetch_add(&R->next, 1); if (t >= R->nlist || atomic_load(&R->fail)) break;
        const int64_t sz = R->list[t][0], sy = R->list[t][1], sx = R->list[t][2];
        memset(shard, R->pair == 2 ? RV_UNKNOWN : PHASE_UNKNOWN, (size_t)512 * 512 * 512);
        praster_job J = {R->z, R->shape, R->ax, R->bp, R->halo, R->blk, {sz * 512, sy * 512, sx * 512}, shard, 0, {512 / R->blk, 512 / R->blk, 512 / R->blk}, 0,
                         PTHREAD_MUTEX_INITIALIZER, R->winding, R->pf, R->fshape, R->pair};
        pthread_t th[64];
        for (int k = 0; k < R->nt; k++) pthread_create(&th[k], nullptr, praster_worker, &J);
        for (int k = 0; k < R->nt; k++) pthread_join(th[k], nullptr);
        if (J.nonempty && pooled) phase_pool2(shard, pooled);
        if (J.nonempty && z3w_write_shard(R->w, sz, sy, sx, pooled ? pooled : shard, R->nt)) { fprintf(stderr, "%s\n", z3w_error()); atomic_store(&R->fail, 1); break; }
        pthread_mutex_lock(&R->mu); R->nonempty += J.nonempty; R->done++;
        fprintf(stderr, "\rshard %lld,%lld,%lld (%zu of %zu): %zu blocks with codes so far, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, R->done, R->nlist, R->nonempty, now() - R->t0);
        pthread_mutex_unlock(&R->mu);
    }
    free(shard); free(pooled); return nullptr;
}
static int cmd_phase_raster(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256] [--pool 2] [--resume 1] [--winding 1] [--gs 4]\n"); return 2; }
    const int pool = atoi(opt(argc, argv, "--pool", "1")), resume = atoi(opt(argc, argv, "--resume", "0"));   /* --pool 2: write only the 9.6 um (2x circular-mean) phase; --resume: skip written shards */
    const int winding = atoi(opt(argc, argv, "--winding", "0"));   /* the winding mod 14 turns (band_winding) instead of the 1-turn phase */
    if (pool != 1 && pool != 2) { fprintf(stderr, "phase-raster: --pool 1|2\n"); return 2; }
    const int gs = atoi(opt(argc, argv, "--gs", "2")), pf = gs / 2;   /* native voxels per output grid voxel: 2 the codes' own 4.8 um grid, 4 the codes pooled 2x (9.6 um) */
    if (gs != 2 && gs != 4) { fprintf(stderr, "phase-raster: --gs 2|4\n"); return 2; }
    if (gs == 4 && pool != 1) { fprintf(stderr, "phase-raster: --gs 4 writes the 9.6 um grid itself (no --pool)\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75")), (float)gs};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "phase-raster: cannot load axis\n"); return 1; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int64_t fshape[3] = {m->shape[0], m->shape[1], m->shape[2]};
    const int64_t shape[3] = {(fshape[0] + pf - 1) / pf, (fshape[1] + pf - 1) / pf, (fshape[2] + pf - 1) / pf};   /* the output grid */
    const int blk = atoi(opt(argc, argv, "--block", "256")), nth = atoi(opt(argc, argv, "--threads", "4"));
    if (512 % blk) { fprintf(stderr, "phase-raster: --block must divide 512\n"); return 2; }
    int64_t lo[3] = {0, 0, 0}, hi[3] = {shape[0], shape[1], shape[2]};
    const char *bx = opt(argc, argv, "--box", nullptr);
    if (bx) { long long v[6]; if (sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return usage(); for (int d = 0; d < 3; d++) { lo[d] = v[d] / gs; hi[d] = (v[d] + v[3 + d] + gs - 1) / gs; } }
    const double um = 2.4 * gs * pool; char lv[32]; z3w_level_name(um, lv, sizeof lv);
    char dir[1400]; snprintf(dir, sizeof dir, "%s/%s", argv[4], lv);
    const int pair_attr = atoi(opt(argc, argv, "--pair", "0"));
    if (pair_attr == 2 && (pool != 1 || winding)) { fprintf(stderr, "phase-raster: --pair 2 writes side labels on the computed grid (no --pool 2 / --winding)\n"); return 2; }
    char attrs[700]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s%s%s\",\"codes\":\"%s/%s\",\"radius\":%g,\"span\":%g}}",
                                   pair_attr == 2 ? "recto / verso side (src/band.h band_field_side_pair: 0 unknown, 1 verso, 2 recto; wrap k .. k + 0.5 recto, k + 0.5 .. k + 1 verso)"
                                   : winding ? "band field winding mod 14 turns x 18 (src/band.h band_winding: 0..251, 255 unknown)" : "band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown)",
                                   pool == 2 ? ", pooled 2x (circular mean of the known codes)" : gs == 4 ? ", computed on the codes pooled 2x (9.6 um grid)" : "",
                                   pair_attr == 1 ? ", pair rule (band_field_phase_pair: the nearest sheets of opposite winding parity)" : "", argv[2], argv[3], bp.radius, bp.span);
    const int64_t oshape[3] = {(shape[0] + pool - 1) / pool, (shape[1] + pool - 1) / pool, (shape[2] + pool - 1) / pool};
    z3w *w = z3w_create(dir, oshape, 512 / pool, 0, pair_attr == 2 ? RV_UNKNOWN : PHASE_UNKNOWN, attrs);   /* a 4.8 shard of 512 -> one output shard */
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    if (z3w_write_group(argv[4], &um, 1, "ufsm-phase", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    /* the shards to compute (--resume skips written ones), --shards S of them at a time, each over --threads block threads: within one
       512^3 shard at most 8 blocks of 256 run, so one shard at a time left most cores idle (Paris4 at 9.6 um: 54 min on the desk) */
    const int nsh = atoi(opt(argc, argv, "--shards", "1")) < 1 ? 1 : atoi(opt(argc, argv, "--shards", "1")) > 64 ? 64 : atoi(opt(argc, argv, "--shards", "1"));
    size_t skipped = 0, nlist = 0, cap = 0; int64_t (*list)[3] = nullptr;
    for (int64_t sz = lo[0] / 512; sz * 512 < hi[0]; sz++) for (int64_t sy = lo[1] / 512; sy * 512 < hi[1]; sy++) for (int64_t sx = lo[2] / 512; sx * 512 < hi[2]; sx++) {
        if (resume) { char sp[1600]; snprintf(sp, sizeof sp, "%s/c/%lld/%lld/%lld", dir, (long long)sz, (long long)sy, (long long)sx); if (access(sp, F_OK) == 0) { skipped++; continue; } }
        if (nlist == cap) { cap = cap ? 2 * cap : 256; list = realloc(list, cap * sizeof *list); }
        list[nlist][0] = sz; list[nlist][1] = sy; list[nlist][2] = sx; nlist++;
    }
    /* --pair 1: band_field_phase_pair (the two nearest sheets of opposite winding parity; no radial inside / outside, no span) */
    const int pair = atoi(opt(argc, argv, "--pair", "0"));
    if (pair && winding) { fprintf(stderr, "phase-raster: --pair writes the 1-turn phase (no --winding)\n"); return 2; }
    praster_run R = {.z = z, .shape = shape, .ax = &ax, .bp = bp, .halo = (int)ceilf((pair == 2 ? 2 * bp.radius : (pair ? 0 : bp.span) + bp.radius) / (float)gs) + 2, .blk = blk, .winding = winding, .pf = pf, .pair = pair,
                     .fshape = fshape, .w = w, .pool = pool, .nt = nth < 1 ? 1 : nth > 64 ? 64 : nth, .list = list, .nlist = nlist, .t0 = now(), .mu = PTHREAD_MUTEX_INITIALIZER};
    pthread_t sth[64];
    for (int k = 1; k < nsh; k++) pthread_create(&sth[k], nullptr, praster_shards, &R);
    praster_shards(&R);
    for (int k = 1; k < nsh; k++) pthread_join(sth[k], nullptr);
    free(list);
    if (R.fail) return 1;
    const size_t nonempty = R.nonempty, nshards = nlist;
    if (z3w_close(w)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    fprintf(stderr, "\nphase-raster: %zu shards (%zu already written), %zu blocks with codes, %.0fs -> %s\n", nshards, skipped, nonempty, now() - R.t0, dir);
    z3_close(z);
    return 0;
}

/* ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis umbilicus.json] [--radius 80] [--span 75] [--threads 8] [--side side.raw]
   [--phase phase.raw]
   Band field (src/band.h) of a native-voxel box (even origin and size) from a winding_mod14 raster level (label grid, 2 native
   voxels per label voxel): writes the band per label voxel, (nz/2) x (ny/2) x (nx/2) uint8, 0..251 or 255 unknown. */
/* bytes this process may still allocate: MemAvailable, capped by its cgroup's memory.max minus memory.current (e.g. inside
   ~/ufsm-data/capped.sh's ufsm.slice); 0 when unknown */
static size_t mem_available(void) {
    size_t avail = 0; char line[512];
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) { while (fgets(line, sizeof line, f)) { unsigned long long kb; if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { avail = (size_t)kb << 10; break; } } fclose(f); }
    f = fopen("/proc/self/cgroup", "r");
    if (f) {
        if (fgets(line, sizeof line, f) && !strncmp(line, "0::", 3)) {
            line[strcspn(line, "\n")] = 0;
            for (char *dir = line + 3; *dir; ) {   /* this cgroup and its ancestors: the tightest max - current */
                char pm[700], pc[700]; snprintf(pm, sizeof pm, "/sys/fs/cgroup%s/memory.max", dir); snprintf(pc, sizeof pc, "/sys/fs/cgroup%s/memory.current", dir);
                FILE *a = fopen(pm, "r"), *b = fopen(pc, "r"); unsigned long long mx = 0, cur = 0; int lim = 0;
                if (a) { lim = fscanf(a, "%llu", &mx) == 1; fclose(a); }
                if (b) { if (fscanf(b, "%llu", &cur) != 1) cur = 0; fclose(b); }
                if (lim && mx > cur && (!avail || mx - cur < avail)) avail = (size_t)(mx - cur);
                char *sl = strrchr(dir, '/'); if (!sl || sl == dir) break; *sl = 0;
            }
        }
        fclose(f);
    }
    return avail;
}
static int cmd_band(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis A] [--radius 80] [--span 75]\n"); return 2; }
    long long b[6];
    if (sscanf(argv[4], "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) return usage();
    for (int d = 0; d < 6; d++) if (b[d] % 2 || (d >= 3 && b[d] <= 0)) { fprintf(stderr, "band: box must be even and positive\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75")), 0};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "band: cannot load axis\n"); return 1; }
    const int halo = (int)ceilf((bp.radius + bp.span) / 2.f) + 2;
    int64_t o[3], n[3]; int ni[3];
    for (int d = 0; d < 3; d++) { o[d] = b[d] / 2 - halo; n[d] = b[3 + d] / 2 + 2 * halo; ni[d] = (int)n[d]; }
    store *s = store_open(argv[2]);
    z3 *z = z3_open(s, argv[3], nullptr);
    if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    const size_t N = (size_t)n[0] * n[1] * n[2];
    const char *side_path = opt(argc, argv, "--side", nullptr);   /* also the recto / verso side per label voxel (band.h SIDE_*) */
    const char *phase_path = opt(argc, argv, "--phase", nullptr);   /* and / or the winding phase (band.h PHASE_*) */
    const char *ff_dir = opt(argc, argv, "--fit-fields", nullptr);   /* and / or the winding fitter's input fields (oracle) */
    const int pri = arg_index(argc, argv, "--phase-raster");   /* with --fit-fields: read the phase from ufsm phase-raster ROOT KEY (large boxes) */
    if (ff_dir && pri > 0 && pri + 2 < argc) {
        const int ffh2 = atoi(opt(argc, argv, "--fit-h", "8"));
        if (ffh2 < 2 || ffh2 % 2 || b[3] % ffh2 || b[4] % ffh2 || b[5] % ffh2) { fprintf(stderr, "band: --fit-h must be even and divide the box\n"); return 2; }
        z3 *pz = z3_open(store_open(argv[pri + 1]), argv[pri + 2], nullptr); if (!pz) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const int ph = 1; int pn[3]; int64_t po[3], pnn[3];   /* the label box with a one-voxel halo (central differences) */
        for (int d = 0; d < 3; d++) { po[d] = b[d] / 2 - ph; pnn[d] = b[3 + d] / 2 + 2 * ph; pn[d] = (int)pnn[d]; }
        const size_t PN = (size_t)pn[0] * pn[1] * pn[2]; uint8_t *phs = malloc(PN);
        memset(phs, PHASE_UNKNOWN, PN);
        int64_t ro2[3], rn2[3]; const z3_meta *pm = z3_meta_of(pz);
        for (int d = 0; d < 3; d++) { ro2[d] = po[d] < 0 ? 0 : po[d]; const int64_t e = po[d] + pnn[d] < pm->shape[d] ? po[d] + pnn[d] : pm->shape[d]; rn2[d] = e - ro2[d]; }
        if (ro2[0] == po[0] && ro2[1] == po[1] && ro2[2] == po[2] && rn2[0] == pnn[0] && rn2[1] == pnn[1] && rn2[2] == pnn[2]) {
            /* the haloed box inside the raster: read in place (a staging copy doubled the peak: a 3.9 cm slab needed ~70 GB) */
            if (z3_read(pz, ro2, rn2, phs, 8)) { fprintf(stderr, "band: phase read failed: %s\n", z3_error()); return 1; }
        } else {
            uint8_t *tmp = malloc((size_t)rn2[0] * rn2[1] * rn2[2]);
            if (z3_read(pz, ro2, rn2, tmp, 8)) { fprintf(stderr, "band: phase read failed: %s\n", z3_error()); return 1; }
            for (int64_t zz = 0; zz < rn2[0]; zz++) for (int64_t yy = 0; yy < rn2[1]; yy++)
                memcpy(phs + ((size_t)(zz + ro2[0] - po[0]) * pnn[1] + (yy + ro2[1] - po[1])) * pnn[2] + (ro2[2] - po[2]), tmp + ((size_t)zz * rn2[1] + yy) * rn2[2], (size_t)rn2[2]);
            free(tmp);
        }
        z3_close(pz);
        axis axr = {0};
        if (axis_load(&axr, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "band: cannot load axis\n"); return 1; }
        const double t0p = now();
        if (band_fit_fields(phs, nullptr, pn, ph, b, ffh2, &axr, ff_dir)) { fprintf(stderr, "band: cannot write the fit fields to %s\n", ff_dir); return 1; }
        fprintf(stderr, "band: fit fields from the stored phase in %.1fs -> %s\n", now() - t0p, ff_dir);
        free(phs);
        return 0;
    }
    const int ffh = atoi(opt(argc, argv, "--fit-h", "8"));   /* field voxel edge, native voxels (even) */
    if (ff_dir && (ffh < 2 || ffh % 2 || b[3] % ffh || b[4] % ffh || b[5] % ffh)) { fprintf(stderr, "band: --fit-h must be even and divide the box\n"); return 2; }
    {   /* the whole haloed box in memory: ~24 bytes per label voxel (5 arrays + the band field's distance maps); refuse rather than
           drive the 24 GB WSL VM into swap and a restart (block B, 1.2e9 voxels: 27 GB). Big boxes: ufsm phase-raster (blockwise),
           then band --fit-fields --phase-raster */
        const double need = 24.0 * (double)N; const size_t av = mem_available();
        if (av && need > 0.8 * (double)av) { fprintf(stderr, "band: the %.2fe9-voxel box needs ~%.1f GB, %.1f GB available; use a smaller box, or ufsm phase-raster (blockwise) and band --fit-fields --phase-raster\n", N / 1e9, need / 1e9, av / 1e9); return 2; }
    }
    uint8_t *codes = calloc(N, 1), *band = malloc(N), *side = side_path ? malloc(N) : nullptr, *phase = phase_path || ff_dir ? malloc(N) : nullptr,
            *spacing = ff_dir ? malloc(N) : nullptr;
    int64_t ro[3], rn[3];   /* the in-bounds part of the haloed box; outside stays empty */
    for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; int64_t e = o[d] + n[d] < m->shape[d] ? o[d] + n[d] : m->shape[d]; rn[d] = e - ro[d]; }
    if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
        uint8_t *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (z3_read(z, ro, rn, tmp, atoi(opt(argc, argv, "--threads", "8")))) { fprintf(stderr, "band: read failed: %s\n", z3_error()); return 1; }
        for (int64_t zz = 0; zz < rn[0]; zz++) for (int64_t yy = 0; yy < rn[1]; yy++)
            memcpy(codes + ((size_t)(zz + ro[0] - o[0]) * n[1] + (yy + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), tmp + ((size_t)zz * rn[1] + yy) * rn[2], (size_t)rn[2]);
        free(tmp);
    }
    double *cy = malloc(n[0] * sizeof(double)), *cx = malloc(n[0] * sizeof(double));
    for (int64_t zz = 0; zz < n[0]; zz++) axis_at(&ax, 2.0 * (zz + o[0]) + 0.5, &cy[zz], &cx[zz]);
    double t0 = now();
    if (band_field_side(codes, ni, o, cy, cx, bp, band, side) || (phase && band_field_phase(codes, ni, o, cy, cx, bp, band, phase, spacing))) { fprintf(stderr, "band: out of memory\n"); return 1; }
    if (ff_dir && band_fit_fields(phase, spacing, ni, halo, b, ffh, &ax, ff_dir)) { fprintf(stderr, "band: cannot write the fit fields to %s\n", ff_dir); return 1; }
    const int64_t on[3] = {b[3] / 2, b[4] / 2, b[5] / 2};
    FILE *f = fopen(argv[5], "wb");
    size_t unknown = 0;
    for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++) {
        const uint8_t *row = band + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo;
        for (int64_t xx = 0; xx < on[2]; xx++) unknown += row[xx] == BAND_UNKNOWN;
        if (!f || fwrite(row, 1, (size_t)on[2], f) != (size_t)on[2]) { fprintf(stderr, "band: write failed\n"); return 1; }
    }
    fclose(f);
    if (side) {
        FILE *fs = fopen(side_path, "wb");
        for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++)
            if (!fs || fwrite(side + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo, 1, (size_t)on[2], fs) != (size_t)on[2]) { fprintf(stderr, "band: side write failed\n"); return 1; }
        fclose(fs); free(side);
    }
    if (phase && phase_path) {
        FILE *fp = fopen(phase_path, "wb");
        for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++)
            if (!fp || fwrite(phase + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo, 1, (size_t)on[2], fp) != (size_t)on[2]) { fprintf(stderr, "band: phase write failed\n"); return 1; }
        fclose(fp);
    }
    free(phase); free(spacing);
    fprintf(stderr, "band field %lldx%lldx%lld (halo %d) in %.1fs, unknown %.1f%%\n", (long long)on[0], (long long)on[1], (long long)on[2], halo,
            now() - t0, 100.0 * unknown / ((double)on[0] * on[1] * on[2]));
    free(codes); free(band); free(cy); free(cx); z3_close(z);
    return 0;
}

/* ufsm fit <fields-dir> [--codes ROOT KEY] [--dir 0|1|-1] [--lattice 512,256,128,64,64,32] [--steps 300,...] [--phase-stage 4]
   [--knot-dz 1024] [--quiet] [--out MODEL-DIR] [--winding ZARR-DIR [--winding-step 4] [--um 2.4]] [--slab Z [--overlap 256]]
   [--model MODEL-DIR] [--axis-normal D] [--cpu [--threads N]] [--phase-robust S] [--conf-pow P] [--bend B] [--w-phase W --w-normal W
   --w-density W --w-smooth W]
   The winding fitter (src/fit.h) on a directory of fit fields (ufsm band --fit-fields, predict --fit-fields / --fit-zarr). --slab:
   a global coarse fit plus z-slabs of Z native voxels (a fit set; --out writes it, --model reads a set or a model). --codes: score
   against a winding_mod14 band raster (label grid, 2 native voxels per label voxel, code = 1 + round(18 q) mod 252 with q the mesh
   coordinate, q = U for dir +1): on every labelled voxel of the box U_fit - q must be one constant mod 14 turns. --axis-normal D: a
   local box away from the axis (crushed scrolls): the axis replaced by a vertical line D native voxels from the box centre along the
   mean sheet normal (D < 0: the other side). --cpu: the CPU backend (no GPU; src/fit_terms.h holds the terms both backends fit).
   --phase-robust S: a Cauchy phase residual of scale S turns; --conf-pow P: phase weight conf^P; --bend B: curvature-adaptive
   smoothness; --w-pgrad W [--pgrad-delta D]: phase-gradient matching (src/fit_terms.h); --unwrap-voxels 1 / 0 / -1: the phase
   stage's integer init by voxels / lattice nodes / none (fit_cfg). --wind4-weight W (default 4): the winding mod 4 term when the fields
   carry dz dy dx dq planes (task unwrap2; 0 off). --w-recto W (default 0): recto anchors r^2 (1 - cos 2 pi W) from the fields' r plane
   (predict --fit-fields / --fit-zarr write it). --out also writes MODEL/fit-config.json (the weights and options of the fit). */
/* U (the unwrapped radius in turns; sheets where U - dir theta / 2 pi is an integer) for the scorer: fill(ctx, og, n3, step, codes, U)
   writes U at og + i step (native voxels relative to the box origin) over the n3 grid, NaN where unknown (it may skip the points
   whose code is not 1..252); returns nonzero on failure */
typedef int (*score_fill_fn)(void *ctx, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U);
/* Score U against the band codes (winding_mod14) of a box, streamed in label z-chunks: the gauge (the mode of U - q mod 14), the
   fraction of winding ids right (|e| < 0.5 turn), |e| < 0.08 turn, median / p90 |e|; detail: integer errors by z octile and by radius
   (cy, cx: the axis per native z, box-local, (int)(shp[0] h) + 1 entries; um: the native voxel size); dump: box-local z y x and U - q
   per voxel. cf: a label voxel in native voxels. Returns 0, or 1 on a read failure. */
/* score_codes' per-voxel tally of one block over threads (contiguous z ranges; counts summed and the detail records concatenated in
   thread order: the same totals and record order as one pass) */
typedef struct { float d; uint8_t z, r; } score_rec;
typedef struct {
    const uint8_t *codes; const float *U; int64_t n3[3], o3[3]; double cf, h, um; const long long *org; const int *shp; const float *cy, *cx; int na_ax, detail;
    size_t *hist[64], *fine[64], nd[64], nan[64]; score_rec *dd[64]; size_t ndet[64], dcap[64]; int nt;
} score_job;
enum { SC_NB = 14 * 72, SC_NF = 14 * 1024 };
static void *score_rows(void *arg) {
    score_job *J = ((void **)arg)[0]; const int t = (int)(intptr_t)((void **)arg)[1];
    const int64_t z0 = J->n3[0] * t / J->nt, z1 = J->n3[0] * (t + 1) / J->nt; const double cf = J->cf, h = J->h; const long long *org = J->org; const int *shp = J->shp;
    size_t *hist = J->hist[t], *fine = J->fine[t], nd = 0, nan = 0;
    for (int64_t zz = z0; zz < z1; zz++) for (int64_t yy = 0; yy < J->n3[1]; yy++) for (int64_t xx = 0; xx < J->n3[2]; xx++) {
        const size_t i = ((size_t)zz * J->n3[1] + yy) * J->n3[2] + xx; const uint8_t cd = J->codes[i];
        if (cd < 1 || cd > 252) continue;
        const double pz = cf * (zz + J->o3[0] + 0.25) - org[0], py = cf * (yy + J->o3[1] + 0.25) - org[1], px = cf * (xx + J->o3[2] + 0.25) - org[2];
        if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
        if (!(J->U[i] == J->U[i])) { nan++; continue; }
        double d = fmod(J->U[i] - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14; if (d >= 14) d = 0;
        hist[(int)(d / 14 * SC_NB) % SC_NB]++; fine[(int)(d / 14 * SC_NF) % SC_NF]++; nd++;
        if (J->detail) {
            if (J->ndet[t] == J->dcap[t]) { J->dcap[t] = J->dcap[t] ? 2 * J->dcap[t] : 1 << 16; J->dd[t] = realloc(J->dd[t], J->dcap[t] * sizeof(score_rec)); }
            const int zi = (int)fmin(7, fmax(0, pz / (shp[0] * h) * 8));
            const int ai = (int)fmin(J->na_ax - 1, fmax(0, pz)); const double rr = hypot(py - J->cy[ai], px - J->cx[ai]) * J->um;
            J->dd[t][J->ndet[t]++] = (score_rec){(float)d, (uint8_t)zi, (uint8_t)fmin(255, rr / 250)};
        }
    }
    J->nd[t] = nd; J->nan[t] = nan;
    return nullptr;
}
static int score_codes(z3 *z, double cf, const long long org[3], const int shp[3], double h, double um, const float *cy, const float *cx, int detail, FILE *dump,
                       score_fill_fn fill, void *ctx) {
    int64_t lo[3], ln[3]; for (int d = 0; d < 3; d++) { lo[d] = (int64_t)floor(org[d] / cf); ln[d] = (int64_t)(shp[d] * h / cf); }
    enum { NB = SC_NB, NF = SC_NF }; size_t *hist = calloc(NB, sizeof(size_t)), *fine = calloc(NF, sizeof(size_t)), nd = 0, nan = 0;   /* d = U - q mod 14: 1/72 and 1/1024 turn bins */
    score_rec *dd = nullptr; size_t ndet = 0, dcap = 0;
    const int na_ax = (int)(shp[0] * h) + 1;
    /* blocks of up to 128 x 1024 x 1024 label voxels aligned to the codes' 128^3 chunks (z planes one at a time decoded every chunk
       128 times: minutes on a 3.9 cm slab) */
    enum { SBZ = 128, SBY = 1024, SBX = 1024 };
    uint8_t *codes = malloc((size_t)SBZ * SBY * SBX); float *U = malloc((size_t)SBZ * SBY * SBX * 4);
    for (int64_t za = lo[0]; za < lo[0] + ln[0]; za = (za / SBZ + 1) * SBZ)
    for (int64_t ya = lo[1]; ya < lo[1] + ln[1]; ya = (ya / SBY + 1) * SBY)
    for (int64_t xa = lo[2]; xa < lo[2] + ln[2]; xa = (xa / SBX + 1) * SBX) {
        const int64_t ze = (za / SBZ + 1) * SBZ < lo[0] + ln[0] ? (za / SBZ + 1) * SBZ : lo[0] + ln[0];
        const int64_t ye = (ya / SBY + 1) * SBY < lo[1] + ln[1] ? (ya / SBY + 1) * SBY : lo[1] + ln[1];
        const int64_t xe = (xa / SBX + 1) * SBX < lo[2] + ln[2] ? (xa / SBX + 1) * SBX : lo[2] + ln[2];
        const int64_t o3[3] = {za, ya, xa}, n3[3] = {ze - za, ye - ya, xe - xa}; const size_t nb3 = (size_t)n3[0] * n3[1] * n3[2];
        if (z3_read(z, o3, n3, codes, 16)) { fprintf(stderr, "score: code read failed: %s\n", z3_error()); free(codes); free(U); free(hist); free(fine); free(dd); return 1; }
        size_t any = 0; for (size_t i = 0; i < nb3 && !any; i++) any = codes[i] >= 1 && codes[i] <= 252;
        if (!any) continue;
        const double og[3] = {cf * o3[0] - 0.25 * cf - org[0], cf * o3[1] - 0.25 * cf - org[1], cf * o3[2] - 0.25 * cf - org[2]};   /* label voxel i at cf (i + lo + 1 / 4) - org */
        if (fill(ctx, og, n3, cf, codes, U)) { fprintf(stderr, "score: cannot evaluate U on the score grid\n"); free(codes); free(U); free(hist); free(fine); free(dd); return 1; }
        if (!dump) {   /* threads (the dump keeps the serial voxel order) */
            score_job J = {codes, U, {n3[0], n3[1], n3[2]}, {o3[0], o3[1], o3[2]}, cf, h, um, org, shp, cy, cx, na_ax, detail, {0}, {0}, {0}, {0}, {0}, {0}, {0}, 0};
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN); const char *te = getenv("UFSM_FIT_THREADS"); if (te && atoi(te) > 0) ncpu = atoi(te);   /* (the fit's threads) */
            J.nt = (int)(ncpu < 1 ? 1 : ncpu > 16 ? 16 : ncpu); if (J.nt > n3[0]) J.nt = (int)n3[0];
            pthread_t th[64]; void *args[64][2];
            for (int t = 0; t < J.nt; t++) { J.hist[t] = calloc(NB, sizeof(size_t)); J.fine[t] = calloc(NF, sizeof(size_t)); args[t][0] = &J; args[t][1] = (void *)(intptr_t)t; pthread_create(&th[t], nullptr, score_rows, args[t]); }
            for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr);
            for (int t = 0; t < J.nt; t++) {
                for (int k = 0; k < NB; k++) hist[k] += J.hist[t][k];
                for (int k = 0; k < NF; k++) fine[k] += J.fine[t][k];
                nd += J.nd[t]; nan += J.nan[t];
                if (J.ndet[t]) { if (ndet + J.ndet[t] > dcap) { dcap = ndet + J.ndet[t] > 2 * dcap ? ndet + J.ndet[t] : 2 * dcap; dd = realloc(dd, dcap * sizeof *dd); }
                                 memcpy(dd + ndet, J.dd[t], J.ndet[t] * sizeof *dd); ndet += J.ndet[t]; }
                free(J.hist[t]); free(J.fine[t]); free(J.dd[t]);
            }
            continue;
        }
        for (int64_t zz = 0; zz < n3[0]; zz++) for (int64_t yy = 0; yy < n3[1]; yy++) for (int64_t xx = 0; xx < n3[2]; xx++) {
            const size_t i = ((size_t)zz * n3[1] + yy) * n3[2] + xx; const uint8_t cd = codes[i];
            if (cd < 1 || cd > 252) continue;
            const double pz = cf * (zz + o3[0] + 0.25) - org[0], py = cf * (yy + o3[1] + 0.25) - org[1], px = cf * (xx + o3[2] + 0.25) - org[2];
            if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
            if (!(U[i] == U[i])) { nan++; continue; }   /* no U here (a raster's unreached voxel) */
            double d = fmod(U[i] - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14; if (d >= 14) d = 0;
            if (dump) { const float rec[4] = {(float)pz, (float)py, (float)px, (float)d}; if (fwrite(rec, 4, 4, dump) != 4) dump = nullptr; }
            hist[(int)(d / 14 * NB) % NB]++; fine[(int)(d / 14 * NF) % NF]++; nd++;
            if (detail) {   /* d and the voxel's z / radius bins for the breakdown after the gauge is known */
                if (ndet == dcap) { dcap = dcap ? 2 * dcap : 1 << 20; dd = realloc(dd, dcap * sizeof *dd); }
                const int zi = (int)fmin(7, fmax(0, pz / (shp[0] * h) * 8));
                const int ai = (int)fmin(na_ax - 1, fmax(0, pz)); const double rr = hypot(py - cy[ai], px - cx[ai]) * um;
                dd[ndet++] = (score_rec){(float)d, (uint8_t)zi, (uint8_t)fmin(255, rr / 250)};   /* radius in 250 um bins */
            }
        }
    }
    free(codes); free(U);
    size_t best = 0; int bi = 0;
    for (int k = 0; k < NB; k++) { size_t sacc = 0; for (int e = -18; e <= 18; e++) sacc += hist[(k + e + NB) % NB]; if (sacc > best) { best = sacc; bi = k; } }   /* +-0.25 turn window */
    const double mode = (bi + 0.5) * 14.0 / NB;
    size_t right = 0, close = 0; double mean = 0;
    for (int k = 0; k < NF; k++) { double e = (k + 0.5) * 14.0 / NF - mode; e -= 14 * round(e / 14); if (fabs(e) < 0.5) { right += fine[k]; mean += e * fine[k]; } }
    mean = right ? mean / right : 0;   /* the gauge refined to the mean error of the right voxels */
    size_t *ah = calloc(NF / 28 + 1, sizeof(size_t));   /* |e| < 0.5 in 1/1024 turn bins */
    for (int k = 0; k < NF; k++) { double e = (k + 0.5) * 14.0 / NF - mode - mean; e -= 14 * round(e / 14); close += fabs(e) < 0.08 ? fine[k] : 0; if (fabs(e) < 0.5) ah[(int)(fabs(e) * 1024)] += fine[k]; }
    size_t na2 = 0; for (int k = 0; k <= NF / 28; k++) na2 += ah[k];
    double med = 0, p90 = 0; { size_t c = 0; int got = 0; for (int k = 0; k <= NF / 28; k++) { c += ah[k]; if (!got && c * 2 >= na2) { med = (k + 0.5) / 1024; got = 1; } if (c * 10 >= na2 * 9) { p90 = (k + 0.5) / 1024; break; } } }
    printf("fit score: %zu labelled voxels, gauge %.3f, winding ids right %.4f, |error| < 0.08 turn %.4f, |error| median %.3f p90 %.3f turn\n", nd, mode + mean,
           nd ? (double)right / nd : 0, nd ? (double)close / nd : 0, na2 ? med : 0, na2 ? p90 : 0);
    if (nan) printf("  %zu labelled voxels without U (%.4f of those in the box), not scored\n", nan, (double)nan / (nan + nd));
    if (detail && ndet) {
        size_t ie[8][9] = {{0}}, re[64][9] = {{0}};   /* integer error -4..4 by z octile / by 250 um radius bin (up to 16 mm) */
        for (size_t i = 0; i < ndet; i++) {
            double e = dd[i].d - mode - mean; e -= 14 * round(e / 14); const int k = (int)lround(e);
            if (k < -4 || k > 4) continue;
            ie[dd[i].z][k + 4]++; if (dd[i].r < 64) re[dd[i].r][k + 4]++;
        }
        printf("  integer error -2 -1 0 +1 +2 (fractions):\n");
        for (int zb = 0; zb < 8; zb++) { size_t t = 0; for (int k = 0; k < 9; k++) t += ie[zb][k]; if (!t) continue;
            printf("    z octile %d %9zu voxels: %.3f %.3f %.3f %.3f %.3f\n", zb, t, (double)ie[zb][2] / t, (double)ie[zb][3] / t, (double)ie[zb][4] / t, (double)ie[zb][5] / t, (double)ie[zb][6] / t); }
        for (int rb = 0; rb < 64; rb += 2) { size_t t = 0, c[9] = {0}; for (int k = 0; k < 9; k++) { c[k] = re[rb][k] + (rb + 1 < 64 ? re[rb + 1][k] : 0); t += c[k]; } if (t < 1000) continue;
            printf("    radius %5.1f-%4.1f mm %9zu voxels: %.3f %.3f %.3f %.3f %.3f\n", rb * 0.25, rb * 0.25 + 0.5, t, (double)c[2] / t, (double)c[3] / t, (double)c[4] / t, (double)c[5] / t, (double)c[6] / t); }
    }
    free(dd); free(ah); free(hist); free(fine);
    return 0;
}
typedef struct { fit_set_dev *gd; const fit_set *fs; } fit_fill_ctx;   /* gd null: the CPU */
static int fit_fill(void *c, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U) {
    fit_fill_ctx *f = c; return f->gd ? fit_set_grid(f->gd, f->fs, og, n3, step, 1, U) : fit_set_grid_cpu(f->fs, og, n3, step, 1, codes, U);
}

static int cmd_fit(int argc, char **argv) {
    const char *dir = argv[2];
    fit_cfg c = fit_cfg_default();
    c.verbose = arg_index(argc, argv, "--quiet") < 0;
    c.dir = atoi(opt(argc, argv, "--dir", "0")); c.unwrap_voxels = atoi(opt(argc, argv, "--unwrap-voxels", "1"));
    c.cpu = arg_index(argc, argv, "--cpu") > 0; c.threads = atoi(opt(argc, argv, "--threads", "0")); fit_set_threads(c.threads);
    c.phase_robust = (float)atof(opt(argc, argv, "--phase-robust", "0")); c.conf_pow = (float)atof(opt(argc, argv, "--conf-pow", "1"));
    c.bend = (float)atof(opt(argc, argv, "--bend", "0"));
    c.w_pgrad = (float)atof(opt(argc, argv, "--w-pgrad", "0")); c.pgrad_delta = (float)atof(opt(argc, argv, "--pgrad-delta", "0.25"));
    c.w_phase = (float)atof(opt(argc, argv, "--w-phase", "1")); c.w_normal = (float)atof(opt(argc, argv, "--w-normal", "3"));
    c.w_density = (float)atof(opt(argc, argv, "--w-density", "1")); c.w_smooth = (float)atof(opt(argc, argv, "--w-smooth", "0.1"));
    c.w_wind4 = (float)atof(opt(argc, argv, "--wind4-weight", "4"));   /* the winding mod 4 term of task unwrap2 fields (dz dy dx dq planes; 0 off) */
    c.w_recto = (float)atof(opt(argc, argv, "--w-recto", "0"));        /* recto anchors (fields with an r plane; 0 off) */
    if (!(c.w_wind4 >= 0) || !(c.w_recto >= 0)) { fprintf(stderr, "fit: --wind4-weight and --w-recto must be >= 0\n"); return 2; }
    if (!(c.w_pgrad >= 0) || !(c.pgrad_delta > 0) || c.unwrap_voxels < -1 || c.unwrap_voxels > 1 || !(c.phase_robust >= 0) || !(c.conf_pow > 0) || !(c.bend >= 0) || !(c.w_phase >= 0) || !(c.w_normal >= 0) || !(c.w_density >= 0) || !(c.w_smooth >= 0)) {
        fprintf(stderr, "fit: --phase-robust, --bend and the weights must be >= 0, --conf-pow and --pgrad-delta > 0, --unwrap-voxels -1, 0 or 1\n"); return 2;
    }
    c.uw_split = atof(opt(argc, argv, "--uw-split", "0")); c.uw_tile = atoi(opt(argc, argv, "--uw-tile", "256")); c.uw_pdv = atof(opt(argc, argv, "--uw-pdv", "0")); c.uw_min_conf = atof(opt(argc, argv, "--uw-min-conf", "0.3"));
    c.core = (float)atof(opt(argc, argv, "--core", "3"));   /* no data terms within core x pitch of the axis */
    c.analytic = atoi(opt(argc, argv, "--analytic", "1")); c.packed_um = atof(opt(argc, argv, "--packed-um", "0")); c.batch = atoi(opt(argc, argv, "--batch", "2000000")); c.knot_dz = atoi(opt(argc, argv, "--knot-dz", "1024")); c.phase_stage = atoi(opt(argc, argv, "--phase-stage", "4"));
    if (*opt(argc, argv, "--lattice", "")) { c.nstage = 0; char *t = strdup(opt(argc, argv, "--lattice", "")); for (char *q = strtok(t, ","); q && c.nstage < 8; q = strtok(nullptr, ",")) c.lattice[c.nstage++] = atoi(q); free(t); }
    if (*opt(argc, argv, "--steps", "")) { int k = 0; char *t = strdup(opt(argc, argv, "--steps", "")); for (char *q = strtok(t, ","); q && k < 8; q = strtok(nullptr, ",")) c.steps[k++] = atoi(q); for (; k < c.nstage; k++) c.steps[k] = c.steps[k - 1]; free(t); }
    if (!c.cpu) nn_init(atoi(opt(argc, argv, "--gpu", "0")));
    const double t0 = now();
    const char *min = opt(argc, argv, "--model", nullptr);   /* score / export a saved model or set instead of fitting */
    const int slab = atoi(opt(argc, argv, "--slab", "0")), overlap = atoi(opt(argc, argv, "--overlap", "256"));
    fit_fields_buf fb; memset(&fb, 0, sizeof fb);
    struct timespec tl0, tl1; clock_gettime(CLOCK_MONOTONIC, &tl0);
    if (fit_fields_load(dir, 0, 0, min || slab > 0 ? 0 : 1, &fb)) { fprintf(stderr, "fit: cannot load the fields in %s\n", dir); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &tl1);
    if (c.verbose) fprintf(stderr, "fit: fields %dx%dx%d loaded in %.1fs\n", fb.shape[0], fb.shape[1], fb.shape[2], (double)(tl1.tv_sec - tl0.tv_sec) + (tl1.tv_nsec - tl0.tv_nsec) * 1e-9);   /* (slab runs load per slab; a given model needs the geometry only) */
    if (*opt(argc, argv, "--axis-normal", "") && !min && !slab) {
        /* local boxes away from the scroll's axis (crushed scrolls): replace the axis by a vertical line D native voxels from the box centre
           along the box's mean sheet normal (top eigenvector of the material-weighted mean n n^T), so the spiral term is nearly flat */
        const double D = atof(opt(argc, argv, "--axis-normal", "0")); const fit_fields *f = &fb.f; const size_t n = (size_t)f->D * f->H * f->W;
        double T[6] = {0}, sw = 0;   /* zz yy xx zy zx yx */
        for (size_t i = 0; i < n; i++) if (f->w[i]) {
            const double w = f->w[i] / 255.0; sw += w;
            for (int k = 0; k < 3; k++) T[k] += w * f->t[k][i] / 255.0;
            for (int k = 3; k < 6; k++) T[k] += w * (f->t[k][i] / 255.0 - 0.5);
        }
        if (sw <= 0) { fprintf(stderr, "fit: --axis-normal: no material in the box\n"); return 1; }
        const double M[3][3] = {{T[0], T[3], T[4]}, {T[3], T[1], T[5]}, {T[4], T[5], T[2]}};
        double v[3] = {0.3, 0.7, 0.6};
        for (int it = 0; it < 100; it++) {
            double u[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) u[a] += M[a][b] * v[b];
            const double l = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (l < 1e-12) break; for (int a = 0; a < 3; a++) v[a] = u[a] / l;
        }
        double ny = v[1], nx = v[2]; const double lyx = sqrt(ny * ny + nx * nx);
        if (lyx < 0.3) fprintf(stderr, "fit: --axis-normal: the mean normal (%.2f %.2f %.2f) is mostly along z\n", v[0], v[1], v[2]);
        if (lyx > 1e-6) { ny /= lyx; nx /= lyx; } else { ny = 1; nx = 0; }
        const double cy = f->H * f->h / 2, cx = f->W * f->h / 2; const int na = (int)(f->D * f->h) + 1;
        for (int z = 0; z < na; z++) { fb.cy[z] = (float)(cy - D * ny); fb.cx[z] = (float)(cx - D * nx); }
        fprintf(stderr, "fit: --axis-normal: mean normal z %.3f y %.3f x %.3f; axis at box-local (y %.1f, x %.1f), %.0f native voxels away\n",
                v[0], v[1], v[2], cy - D * ny, cx - D * nx, D);
    }
    c.um = fb.um;   /* default lattices and knots are physical: given in 2.4 um voxels, scaled to the fields' native voxels */
    if (!*opt(argc, argv, "--knot-dz", "")) c.knot_dz = fit_scale_L(1024, 2.4 / fb.um);
    if (!*opt(argc, argv, "--lattice", "")) for (int k = 0; k < c.nstage; k++) c.lattice[k] = fit_scale_L(c.lattice[k], 2.4 / fb.um);
    fit_set *fs = nullptr;
    if (min) fs = fit_set_load(min);
    else if (slab > 0) fs = fit_run_slabs(dir, &c, slab, overlap);
    else { fs = calloc(1, sizeof *fs); fs->global = fit_run(&fb.f, &c); if (!fs->global) { free(fs); fs = nullptr; } }
    if (!fs) { fprintf(stderr, "fit: no model%s%s\n", min ? " in " : "", min ? min : ""); return 1; }
    const fit_model *m = fs->global;
    if (!min && !slab) for (int d = 0; d < 3; d++) fs->global->origin[d] = fb.origin[d];
    if (!min) { fs->global->um = fb.um; for (int k = 0; k < fs->n; k++) fs->slab[k]->um = fb.um; }
    const long long org[3] = {m->origin[0], m->origin[1], m->origin[2]};
    const int shp[3] = {fb.shape[0], fb.shape[1], fb.shape[2]}; const double h = fb.h;
    fprintf(stderr, "fit: dir %+d, %d slab model(s), global lattice %d (%dx%dx%d), loss %.5f in %.1fs\n", m->dir, fs->n, m->L, m->lz, m->ly, m->lx, m->loss[0], now() - t0);
    const char *mout = opt(argc, argv, "--out", nullptr);
    if (mout && (fs->n ? fit_set_save(fs, mout) : fit_save(fs->global, mout))) { fprintf(stderr, "fit: cannot save %s\n", mout); return 1; }
    if (mout && !min) {   /* the fit's options beside the model (reproducibility) */
        char cp[1400]; snprintf(cp, sizeof cp, "%s/fit-config.json", mout); FILE *cf = fopen(cp, "w");
        if (!cf) { fprintf(stderr, "fit: cannot save %s\n", cp); return 1; }
        fprintf(cf, "{\"version\":1,\"w_phase\":%.9g,\"w_normal\":%.9g,\"w_density\":%.9g,\"w_smooth\":%.9g,\"w_pgrad\":%.9g,\"pgrad_delta\":%.9g,\"w_wind4\":%.9g,\"w_recto\":%.9g,"
                    "\"phase_robust_turns\":%.9g,\"conf_pow\":%.9g,\"bend\":%.9g,\"unwrap_voxels\":%d,\"uw_min_conf\":%.9g,\"phase_stage\":%d,\"slab\":%d,\"cpu\":%d,\"native_um\":%.9g,\"lattice\":[",
                c.w_phase, c.w_normal, c.w_density, c.w_smooth, c.w_pgrad, c.pgrad_delta, c.w_wind4, c.w_recto, c.phase_robust, c.conf_pow, c.bend, c.unwrap_voxels, c.uw_min_conf,
                c.phase_stage, slab, c.cpu, c.um);
        for (int k = 0; k < c.nstage; k++) fprintf(cf, "%s%d", k ? "," : "", c.lattice[k]);
        fprintf(cf, "]}\n"); if (fclose(cf)) { fprintf(stderr, "fit: cannot save %s\n", cp); return 1; }
    }
    const char *wout = opt(argc, argv, "--winding", nullptr);
    if (wout) {   /* the per-voxel winding number W * 256 (uint16, exact volcomp typed chunks) on a grid of --winding-step native voxels */
        const int ws = atoi(opt(argc, argv, "--winding-step", "4")); const double um = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : fb.um;
        const int64_t gs[3] = {(int64_t)ceil(shp[0] * h / ws), (int64_t)ceil(shp[1] * h / ws), (int64_t)ceil(shp[2] * h / ws)};
        char lv[32], ld[1400], at[600]; z3w_level_name(um * ws, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", wout, lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding coordinate W * 256 (uint16; W = U - dir theta / 2 pi, sheets at integer W)\",\"dir\":%d,\"origin_zyx\":[%lld,%lld,%lld],\"step\":%d}}",
                 m->dir, org[0], org[1], org[2], ws);
        z3w *zw = z3w_create_typed(ld, gs, 128, 1, 0.f, 0.0, at);
        if (!zw || z3w_write_group(wout, (const double[]){um * ws}, 1, "ufsm-winding", at)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        uint16_t *sb = malloc((size_t)128 * 128 * 128 * 2); float *wv = malloc((size_t)128 * 128 * 128 * 4);
        fit_set_dev *gd = nullptr; if (!c.cpu && !(gd = fit_set_upload(fs))) return 1;
        for (int64_t sz = 0; sz * 128 < gs[0]; sz++) for (int64_t sy = 0; sy * 128 < gs[1]; sy++) for (int64_t sx = 0; sx * 128 < gs[2]; sx++) {
            const int64_t n3[3] = {gs[0] - sz * 128 < 128 ? gs[0] - sz * 128 : 128, gs[1] - sy * 128 < 128 ? gs[1] - sy * 128 : 128, gs[2] - sx * 128 < 128 ? gs[2] - sx * 128 : 128};
            const double o3[3] = {(double)sz * 128 * ws, (double)sy * 128 * ws, (double)sx * 128 * ws};
            if (gd ? fit_set_grid(gd, fs, o3, n3, ws, 0, wv) : fit_set_grid_cpu(fs, o3, n3, ws, 0, nullptr, wv)) { fprintf(stderr, "fit: no device memory for the winding grid\n"); return 1; }
            memset(sb, 0, (size_t)128 * 128 * 128 * 2);
            for (int64_t a = 0; a < n3[0]; a++) for (int64_t b = 0; b < n3[1]; b++) for (int64_t c2 = 0; c2 < n3[2]; c2++) {
                const double v = (double)wv[((size_t)a * n3[1] + b) * n3[2] + c2] * 256; sb[((size_t)a * 128 + b) * 128 + c2] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : lround(v));
            }
            if (z3w_write_shard(zw, sz, sy, sx, (const uint8_t *)sb, 8)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        }
        fit_set_dev_free(gd); free(wv);
        z3w_close(zw); free(sb);
        fprintf(stderr, "fit: winding %lldx%lldx%lld (step %d) written to %s\n", (long long)gs[0], (long long)gs[1], (long long)gs[2], ws, wout);
    }
    const int ci = arg_index(argc, argv, "--codes");
    if (ci > 0 && ci + 2 < argc) {   /* score against the band codes of the box, streamed in label z-chunks (U on the GPU) */
        z3 *z = z3_open(store_open(argv[ci + 1]), argv[ci + 2], nullptr);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const double cf = atof(argv[ci + 2]) > 0 ? atof(argv[ci + 2]) / fb.um : 2;   /* label voxel in native voxels (the codes' key is its um: 4.8) */
        FILE *dump = *opt(argc, argv, "--score-dump", "") ? fopen(opt(argc, argv, "--score-dump", ""), "wb") : nullptr;   /* box-local z y x, U - q mod 14 per voxel */
        fit_set_dev *gd = nullptr; if (!c.cpu && !(gd = fit_set_upload(fs))) return 1;
        fit_fill_ctx fc = {gd, fs};
        const int rc = score_codes(z, cf, org, shp, h, fb.um, fb.cy, fb.cx, arg_index(argc, argv, "--score-detail") > 0, dump, fit_fill, &fc);
        fit_set_dev_free(gd); z3_close(z); if (dump) fclose(dump);
        if (rc) return 1;
    }
    fit_set_free(fs); fit_fields_free(&fb);
    return 0;
}

/* ufsm wind-score <sweep-dir> <codes-root> <codes-key> [--score-detail 1] [--score-dump FILE]
   The fitter's --codes score (score_codes) of a winding raster written by predict --cond-sweep: <sweep-dir>/U.f32 (float32 [z][y][x]
   over the box, NaN where the sweep did not reach) and sweep.json (shape, origin_zyx and um of the box's level grid, scale = level
   voxels per native 2.4 um voxel, axis). U is read at the voxel holding each label voxel; the same gauge, ids-right, |error| and
   by-z / by-radius integer errors as fit --codes, plus the labelled voxels the sweep did not reach. */
typedef struct { const float *U; int64_t n[3]; } raster_fill_ctx;
static int raster_fill(void *c, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U) {
    const raster_fill_ctx *r = c;
    for (int64_t a = 0; a < n3[0]; a++) for (int64_t b = 0; b < n3[1]; b++) for (int64_t x = 0; x < n3[2]; x++) {
        const int64_t iz = (int64_t)floor(og[0] + a * step), iy = (int64_t)floor(og[1] + b * step), ix = (int64_t)floor(og[2] + x * step);
        U[((size_t)a * n3[1] + b) * n3[2] + x] = iz < 0 || iy < 0 || ix < 0 || iz >= r->n[0] || iy >= r->n[1] || ix >= r->n[2] ? NAN : r->U[((size_t)iz * r->n[1] + iy) * r->n[2] + ix];
    }
    return 0;
}
static int cmd_wind_score(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wind-score <sweep-dir> <codes-root> <codes-key> [--score-detail 1] [--score-dump FILE]\n"); return 2; }
    char p[1400]; snprintf(p, sizeof p, "%s/sweep.json", argv[2]);
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "wind-score: cannot read %s\n", p); return 1; }
    char jb[65536]; const size_t jn = fread(jb, 1, sizeof jb - 1, f); fclose(f); jb[jn] = 0;
    json *j = json_parse(jb, jn); if (!j) { fprintf(stderr, "wind-score: bad %s\n", p); return 1; }
    long long org[3]; int shp[3];
    for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin_zyx"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
    const double um = json_num(json_get(j, "um"), 9.6), scale = json_num(json_get(j, "scale"), 4);
    axis ax = {0};
    if (axis_load(&ax, json_str(json_get(j, "axis"), ""))) { fprintf(stderr, "wind-score: cannot load the axis named in %s\n", p); json_free(j); return 1; }
    json_free(j);
    const size_t nv = (size_t)shp[0] * shp[1] * shp[2]; float *Ur = malloc(nv * 4);
    snprintf(p, sizeof p, "%s/U.f32", argv[2]); f = fopen(p, "rb");
    if (!Ur || !f || fread(Ur, 4, nv, f) != nv) { fprintf(stderr, "wind-score: cannot read %s (%zu values)\n", p, nv); if (f) fclose(f); return 1; }
    fclose(f);
    const int na = shp[0] + 1; float *cy = malloc((size_t)na * 4), *cx = malloc((size_t)na * 4);   /* the axis per box z, box-local level voxels */
    for (int z = 0; z < na; z++) { double y, x; axis_at(&ax, (org[0] + z) * scale, &y, &x); cy[z] = (float)(y / scale - org[1]); cx[z] = (float)(x / scale - org[2]); }
    z3 *zc = z3_open(store_open(argv[3]), argv[4], nullptr); if (!zc) { fprintf(stderr, "wind-score: %s\n", z3_error()); return 1; }
    const double cf = atof(argv[4]) > 0 ? atof(argv[4]) / um : 2;
    FILE *dump = *opt(argc, argv, "--score-dump", "") ? fopen(opt(argc, argv, "--score-dump", ""), "wb") : nullptr;
    raster_fill_ctx rc = {Ur, {shp[0], shp[1], shp[2]}};
    const int r = score_codes(zc, cf, org, shp, 1.0, um, cy, cx, atoi(opt(argc, argv, "--score-detail", "0")) > 0, dump, raster_fill, &rc);
    z3_close(zc); if (dump) fclose(dump); free(Ur); free(cy); free(cx);
    return r;
}

/* ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16]
   [--snap ROOT KEY [--snap-range 0.35] [--snap-min 0.5] [--snap-origin z,y,x] | --snap-fields DIR] [--out-um U]
   One tifxyz per wrap of a fitted winding model (ufsm fit --out): wrap k is the level set W = k. Rows every step native voxels in
   z, columns every step / r_k radians of theta over the box's angular range (r_k: the wrap's radius at the box centre), each
   vertex the root of W = k along the ray from the umbilicus (src/fit.h fit_ray_root); vertices outside the box, without a root,
   or (--fields) in a field voxel without data are -1. Coordinates are global native voxels. */
static float trilin(const uint8_t *v, const int64_t n[3], double z, double y, double x);
/* ufsm wraps --snap: each vertex moved along its ray from the axis to the recto maximum within +- range pitches, read from a
   recto probability array on the model's native grid (array voxel = native - ro), when that maximum is >= pmin; then the
   displacements are median-filtered over 5 x 5 vertices so a stray peak cannot tear the surface. Tiles of rows x columns read
   only their own box (plus the search range); the vertices of a tile run on threads. */
typedef struct { const fit_model *m; const uint8_t *vol; int64_t o[3], n[3]; const float *px, *py, *pz; float *disp; int cols, r0, r1, c0, c1;
                 double range, pmin; _Atomic int next; double ih, sub; int nearest; } snap_job;   /* array index = (native - o) ih - sub;
                 nearest: the local maximum nearest the vertex (else the highest within the range) */
static double model_pitch(const fit_model *m, double z) {
    double k = z / m->knot_dz; if (k < 0) k = 0; if (k > m->nk - 1) k = m->nk - 1;
    int k0 = (int)k; if (k0 > m->nk - 2) k0 = m->nk - 2 < 0 ? 0 : m->nk - 2;
    return m->nk > 1 ? m->pitch[k0] + (k - k0) * (m->pitch[k0 + 1] - m->pitch[k0]) : m->pitch[0];
}
static void *snap_rows(void *arg) {
    snap_job *J = arg; const fit_model *m = J->m; const int na = (int)(m->D * m->h) + 1;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * J->cols + c; if (J->pz[i] < 0) continue;
        const double zl = J->pz[i] - m->origin[0], zi = fmin(fmax(zl, 0), na - 1);
        const double cy = m->cy[(int)zi] + m->origin[1], cx = m->cx[(int)zi] + m->origin[2];
        double uy = J->py[i] - cy, ux = J->px[i] - cx; const double ul = hypot(uy, ux); if (ul < 1e-6) continue; uy /= ul; ux /= ul;
        const double R = J->range * model_pitch(m, zl);
#define SNAPV(t_) trilin(J->vol, J->n, (J->pz[i] - J->o[0]) * J->ih - J->sub, (J->py[i] + (t_) * uy - J->o[1]) * J->ih - J->sub, (J->px[i] + (t_) * ux - J->o[2]) * J->ih - J->sub)
        if (J->nearest) {   /* the local maximum nearest the vertex, >= pmin (the sheet the fit is on, not a brighter neighbour) */
            const double st = 0.25; double bd = 1e300, bt = 0; float vm = SNAPV(-R - st), v0 = SNAPV(-R);
            for (double t = -R; t <= R; t += st) {
                const float vp = SNAPV(t + st);
                if (v0 >= vm && v0 > vp && v0 >= J->pmin * 255 && fabs(t) < bd) { bd = fabs(t); bt = t; }
                vm = v0; v0 = vp;
            }
            if (bd < 1e300) J->disp[i] = (float)bt;
            continue;
        }
        float best = -1; double bt = 0;
        for (double t = -R; t <= R; t += 0.5) {
            const float v = SNAPV(t);
            if (v > best) { best = v; bt = t; }
        }
#undef SNAPV
        if (best >= J->pmin * 255) J->disp[i] = (float)bt;
    }
    return nullptr;
}
static int cmp_fl(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
/* mem (with --snap-fields): an in-memory array of shape mn on a grid of cell mh native voxels from mo (cell centres), searched for
   the nearest local maximum instead of a zarr read per tile */
static size_t wraps_snap(z3 *rz, const int64_t ro[3], const fit_model *m, int rows, int cols, float *px, float *py, float *pz, double range, double pmin, int nthreads,
                         const uint8_t *mem, const int mn[3], const long long mo[3], double mh) {
    const size_t nv = (size_t)rows * cols; float *disp = malloc(nv * 4); for (size_t i = 0; i < nv; i++) disp[i] = NAN;
    if (mem) {
        snap_job J = {m, mem, {mo[0], mo[1], mo[2]}, {mn[0], mn[1], mn[2]}, px, py, pz, disp, cols, 0, rows, 0, cols, range, pmin, 0, 1 / mh, 0.5, 1};
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, snap_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    }
    const z3_meta *zm = rz ? z3_meta_of(rz) : nullptr; double pmax = 0; for (int k = 0; k < m->nk; k++) if (m->pitch[k] > pmax) pmax = m->pitch[k];
    const int TR = 64, TC = 512; const double mar = range * pmax + 3;
    for (int r0 = 0; r0 < (mem ? 0 : rows); r0 += TR) for (int c0 = 0; c0 < cols; c0 += TC) {
        const int r1 = r0 + TR < rows ? r0 + TR : rows, c1 = c0 + TC < cols ? c0 + TC : cols;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
            const size_t i = (size_t)r * cols + c; if (pz[i] < 0) continue;
            const double v[3] = {pz[i], py[i], px[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
        }
        if (lo[0] > hi[0]) continue;
        snap_job J = {m, nullptr, {0, 0, 0}, {0, 0, 0}, px, py, pz, disp, cols, r0, r1, c0, c1, range, pmin, 0, 1, 0, 0};
        int ok = 1;
        for (int d = 0; d < 3; d++) {
            const double g = d ? mar : 2;   /* rays are horizontal: the search moves y / x only */
            int64_t a = (int64_t)floor(lo[d] - g) - ro[d], b = (int64_t)ceil(hi[d] + g) + 1 - ro[d];
            if (a < 0) a = 0;
            if (b > zm->shape[d]) b = zm->shape[d];
            if (b <= a) ok = 0;
            J.o[d] = a + ro[d]; J.n[d] = b - a;
        }
        if (!ok) continue;
        uint8_t *vol = malloc((size_t)J.n[0] * J.n[1] * J.n[2]);
        const int64_t ao[3] = {J.o[0] - ro[0], J.o[1] - ro[1], J.o[2] - ro[2]};
        if (!vol || z3_read(rz, ao, J.n, vol, nthreads)) { free(vol); continue; }
        J.vol = vol;
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, snap_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        free(vol);
    }
    size_t moved = 0; float *med = malloc(nv * 4); float win[25];
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
        const size_t i = (size_t)r * cols + c; med[i] = NAN; if (pz[i] < 0) continue;
        int k = 0;
        for (int a = r - 2; a <= r + 2; a++) for (int b = c - 2; b <= c + 2; b++)
            if (a >= 0 && a < rows && b >= 0 && b < cols && !isnan(disp[(size_t)a * cols + b])) win[k++] = disp[(size_t)a * cols + b];
        if (k >= 5) { qsort(win, k, sizeof(float), cmp_fl); med[i] = win[k / 2]; }
    }
    const int na = (int)(m->D * m->h) + 1;
    for (size_t i = 0; i < nv; i++) {
        if (isnan(med[i]) || med[i] == 0) continue;
        const double zl = pz[i] - m->origin[0], zi = fmin(fmax(zl, 0), na - 1);
        const double cy = m->cy[(int)zi] + m->origin[1], cx = m->cx[(int)zi] + m->origin[2];
        double uy = py[i] - cy, ux = px[i] - cx; const double ul = hypot(uy, ux); if (ul < 1e-6) continue;
        py[i] = (float)(py[i] + med[i] * uy / ul); px[i] = (float)(px[i] + med[i] * ux / ul); moved++;
    }
    free(disp); free(med);
    return moved;
}
/* ufsm wraps --fields: per vertex row (z) and angle bin, the radius range [rmin, rmax] of the points around the row's axis point
   whose data-weight cell can be non-zero (4 x 4 cell blocks with a non-zero cell or neighbour, padded by a native voxel; the bin
   and its neighbours). A vertex outside it is dropped anyway, so ray roots outside are not searched for (fit_set_ray_root_b): on a
   2 cm half box ~93% of the rays found no kept vertex (in the masked half, in the void, or off the wrap) at ~30-180 W evaluations
   each. The kept vertices are the same. */
typedef struct { int rows, nb; double t0, bw; float *rmin, *rmax; } wr_polar;
typedef struct { wr_polar *P; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh, step, Z; int na; _Atomic int next; } wr_polar_job;
static void *wr_polar_rows(void *arg) {
    wr_polar_job *J = arg; wr_polar *P = J->P; const fit_model *m = J->m; const int B = 4, H = J->wsh[1], W = J->wsh[2];
    const int Hb = (H + B - 1) / B, Wb = (W + B - 1) / B; uint8_t *blk = malloc((size_t)Hb * Wb);
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < P->rows;) {
        const double z = fmin(r * J->step, J->Z), cy = m->cy[(int)fmin(z, J->na - 1)], cx = m->cx[(int)fmin(z, J->na - 1)];
        const int fz = (int)fmin(z / J->wh, J->wsh[0] - 1); const uint8_t *pl = J->wf + (size_t)fz * H * W;
        float *rn = P->rmin + (size_t)r * P->nb, *rx = P->rmax + (size_t)r * P->nb;
        for (int q = 0; q < P->nb; q++) { rn[q] = 1e30f; rx[q] = -1e30f; }
        memset(blk, 0, (size_t)Hb * Wb);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (pl[(size_t)y * W + x])   /* the cell and its neighbours' blocks */
            for (int a = y - 1; a <= y + 1; a++) for (int b = x - 1; b <= x + 1; b++) if (a >= 0 && b >= 0 && a < H && b < W) blk[(size_t)(a / B) * Wb + b / B] = 1;
        for (int by = 0; by < Hb; by++) for (int bx = 0; bx < Wb; bx++) {
            if (!blk[(size_t)by * Wb + bx]) continue;
            const double y0 = by * B * J->wh - 1 - cy, y1 = (by + 1) * B * J->wh + 1 - cy, x0 = bx * B * J->wh - 1 - cx, x1 = (bx + 1) * B * J->wh + 1 - cx;
            const double ny = y0 > 0 ? y0 : y1 < 0 ? -y1 : 0, nx = x0 > 0 ? x0 : x1 < 0 ? -x1 : 0;
            const double rlo = sqrt(ny * ny + nx * nx) - 1, rhi = sqrt(fmax(y0 * y0, y1 * y1) + fmax(x0 * x0, x1 * x1)) + 1;
            int q0 = 0, q1 = P->nb - 1, full = rlo <= 2;
            double alo = 0, ahi = 0;
            if (!full) {   /* the block's angles around the axis point: its corners' relative to its centre's */
                const double ac = atan2(0.5 * (y0 + y1), 0.5 * (x0 + x1)); double dmin = 0, dmax = 0;
                const double cyv[2] = {y0, y1}, cxv[2] = {x0, x1};
                for (int c = 0; c < 4; c++) { double d = atan2(cyv[c >> 1], cxv[c & 1]) - ac; d -= 2 * M_PI * floor((d + M_PI) / (2 * M_PI)); if (d < dmin) dmin = d; if (d > dmax) dmax = d; }
                alo = ac + dmin; ahi = ac + dmax;
            }
            for (int sh = -1; sh <= 1; sh++) {
                if (!full) {
                    const double lo = (alo + 2 * M_PI * sh - P->t0) / P->bw, hi = (ahi + 2 * M_PI * sh - P->t0) / P->bw;
                    if (hi < 0 || lo > P->nb - 1) continue;
                    q0 = lo < 0 ? 0 : (int)floor(lo); q1 = hi > P->nb - 1 ? P->nb - 1 : (int)floor(hi);
                } else if (sh) continue;
                for (int q = q0; q <= q1; q++) { if (rlo < rn[q]) rn[q] = (float)fmax(rlo, 0); if (rhi > rx[q]) rx[q] = (float)rhi; }
            }
        }
    }
    free(blk);
    return nullptr;
}
static void wr_polar_bounds(const wr_polar *P, int r, double th, double *ua, double *ub) {   /* the bins at th and their neighbours */
    *ua = -1e300; *ub = 1e300; if (!P) return;
    const double t = (th - P->t0) / P->bw; if (!(t >= 1) || t >= P->nb - 1) return;   /* (outside the table: unbounded) */
    const int q = (int)floor(t); double a = 1e300, b = -1e300;
    for (int k = q - 1; k <= q + 1; k++) { const float lo = P->rmin[(size_t)r * P->nb + k], hi = P->rmax[(size_t)r * P->nb + k]; if (lo <= hi) { if (lo < a) a = lo; if (hi > b) b = hi; } }
    *ua = a; *ub = b;   /* (no block: a > b, no root wanted) */
}
typedef struct {   /* ufsm wraps: one wrap's rows over threads */
    const fit_set *fs; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh;
    double step, Z; int na; double th0, dth; int rows, cols, k; double rk;
    float *px, *py, *pz; _Atomic int next; _Atomic size_t valid; const wr_polar *P;
} wraps_job;
static void *wraps_rows(void *arg) {
    wraps_job *J = arg; const fit_model *m = J->m;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->rows;) {
        const double z = fmin(r * J->step, J->Z), cy = m->cy[(int)fmin(z, J->na - 1)], cx = m->cx[(int)fmin(z, J->na - 1)];
        double rprev = J->rk; size_t valid = 0;
        for (int c = 0; c < J->cols; c++) {
            const size_t i = (size_t)r * J->cols + (J->cols - 1 - c);   /* column 0 = the largest theta side (upstream: outermost first) */
            const double th = J->th0 + c * J->dth; double rr;
            J->px[i] = J->py[i] = J->pz[i] = -1;
            double ua, ub; wr_polar_bounds(J->P, r, th, &ua, &ub);
            if (fit_set_ray_root_b(J->fs, z, th, J->k, rprev, ua, ub, &rr)) continue;
            const double y = cy + rr * sin(th), x = cx + rr * cos(th);
            if (J->wf) {
                const int fz = (int)fmin(z / J->wh, J->wsh[0] - 1), fy = (int)fmin(y / J->wh, J->wsh[1] - 1), fx = (int)fmin(x / J->wh, J->wsh[2] - 1);
                if (fy < 0 || fx < 0 || !J->wf[((size_t)fz * J->wsh[1] + fy) * J->wsh[2] + fx]) continue;
            }
            J->px[i] = (float)(x + m->origin[2]); J->py[i] = (float)(y + m->origin[1]); J->pz[i] = (float)(z + m->origin[0]); valid++; rprev = rr;
        }
        atomic_fetch_add(&J->valid, valid);
    }
    return nullptr;
}
typedef struct {
    const fit_set *fs; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh, step, Z; int na; double th0, th1; int rows, k0, k1, nthreads; double um, out_um;
    const char *out; z3 *snapz; const int64_t *snapo; double snap_range, snap_min; const uint8_t *sfa; int sfn[3]; long long sfo[3]; double sfh; int verbose_snap; double Y, X;
    const wr_polar *P; _Atomic int next; _Atomic int nwr; _Atomic size_t nvalid; _Atomic int fail;
} wraps_all;
static int wrap_one(wraps_all *A, int k, int nthreads) {   /* wrap k: its rows on nthreads threads, snapped, written */
    const fit_model *m = A->m; const double Z = A->Z, Y = A->Y, X = A->X, step = A->step; const int na = A->na, rows = A->rows;
    /* the wrap's radius at the box centre: the root along the central ray */
    const double zc = Z / 2, cyc = m->cy[(int)fmin(zc, na - 1)], cxc = m->cx[(int)fmin(zc, na - 1)];
    const double thc = atan2(Y / 2 - cyc, X / 2 - cxc); double rk;
    if (fit_set_ray_root(A->fs, zc, thc, k, hypot(Y / 2 - cyc, X / 2 - cxc), &rk)) rk = hypot(Y / 2 - cyc, X / 2 - cxc);
    const double dth = step / fmax(rk, step);
    const int cols = (int)ceil((A->th1 - A->th0) / dth) + 1;
    const size_t nv = (size_t)rows * cols;
    float *px = malloc(nv * 4), *py = malloc(nv * 4), *pz = malloc(nv * 4);
    wraps_job J = {A->fs, m, A->wf, {A->wsh[0], A->wsh[1], A->wsh[2]}, A->wh, step, Z, na, A->th0, dth, rows, cols, k, rk, px, py, pz, 0, 0, A->P};
    if (nthreads > 1) {
        pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wraps_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    } else wraps_rows(&J);
    const size_t valid = atomic_load(&J.valid);
    if ((A->snapz || A->sfa) && valid) { const size_t mv = wraps_snap(A->snapz, A->snapo, m, rows, cols, px, py, pz, A->snap_range, A->snap_min, nthreads, A->sfa, A->sfn, A->sfo, A->sfh); if (A->verbose_snap) fprintf(stderr, "wraps: w%03d: %zu of %zu vertices snapped\n", k, mv, valid); }
    int rc = 0;
    if (valid > 16) {
        char wd[1400]; snprintf(wd, sizeof wd, "%s/w%03d", A->out, k);
        double wstep = step, wum = A->um;
        if (A->out_um > 0 && fabs(A->out_um - A->um) > 1e-9) {   /* --out-um: coordinates on another grid of the same scan (voxel centres to voxel centres) */
            const double f = A->um / A->out_um;
            for (size_t i = 0; i < nv; i++) if (pz[i] >= 0) { px[i] = (float)((px[i] + 0.5) * f - 0.5); py[i] = (float)((py[i] + 0.5) * f - 0.5); pz[i] = (float)((pz[i] + 0.5) * f - 0.5); }
            wstep = step * f; wum = A->out_um;
        }
        if (tifxyz_write(wd, cols, rows, px, py, pz, wstep, wum, k, "ufsm wraps (fitted winding level set)")) { fprintf(stderr, "wraps: cannot write %s\n", wd); rc = -1; }
        else { atomic_fetch_add(&A->nwr, 1); atomic_fetch_add(&A->nvalid, valid); }
    }
    free(px); free(py); free(pz);
    return rc;
}
static void *wraps_worker(void *arg) {
    wraps_all *A = arg;
    for (int k; (k = A->k0 + atomic_fetch_add(&A->next, 1)) <= A->k1;) if (wrap_one(A, k, 1)) atomic_store(&A->fail, 1);
    return nullptr;
}
static int cmd_wraps(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16]\n"); return 2; }
    fit_set *fs = fit_set_load(argv[2]); if (!fs) { fprintf(stderr, "wraps: cannot load %s\n", argv[2]); return 1; }
    const fit_model *m = fs->global;   /* the box, the axis; rows dispatch to the slab model of their z */
    const double step = atof(opt(argc, argv, "--step", "8")), um = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : fs->global->um;
    const int nthreads = atoi(opt(argc, argv, "--threads", "16")); fit_set_threads(nthreads);   /* (the field loads' core masks) */
    const double out_um = atof(opt(argc, argv, "--out-um", "0"));   /* e.g. 2.4: a 9.6 um fit's wraps on the 2.4 um grid (renders with 2.4 um ink) */
    const int si = arg_index(argc, argv, "--snap");   /* --snap ROOT KEY: a recto probability array on the model's grid (predict output) */
    z3 *snapz = nullptr; int64_t snapo[3] = {0, 0, 0}; const double snap_range = atof(opt(argc, argv, "--snap-range", "0.35")), snap_min = atof(opt(argc, argv, "--snap-min", "0.5"));
    const int verbose_snap = arg_index(argc, argv, "--quiet") < 0;
    if (si > 0) {
        if (si + 2 >= argc) { fprintf(stderr, "wraps: --snap ROOT KEY\n"); return 2; }
        snapz = z3_open(store_open(argv[si + 1]), argv[si + 2], opt(argc, argv, "--cache", nullptr));
        if (!snapz) { fprintf(stderr, "wraps: --snap: %s\n", z3_error()); return 1; }
        const char *so = opt(argc, argv, "--snap-origin", nullptr);   /* else the array's ufsm.origin_zyx (predict writes it) */
        if (so) { long long v[3]; if (sscanf(so, "%lld,%lld,%lld", v, v + 1, v + 2) != 3) { fprintf(stderr, "wraps: --snap-origin z,y,x\n"); return 2; } for (int d = 0; d < 3; d++) snapo[d] = v[d]; }
        else {
            char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[si + 1], argv[si + 2]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
            if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
            const char *q = strstr(jb, "\"origin_zyx\":"); long long v[3];
            if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", v, v + 1, v + 2) != 3) { fprintf(stderr, "wraps: --snap: no origin_zyx in %s (give --snap-origin)\n", jp); return 2; }
            for (int d = 0; d < 3; d++) snapo[d] = v[d];
        }
    }
    /* --snap-fields DIR: snap to the sheet centres of a fit-field set (its pc = 128 + 127 conf cos(phase) peaks at phase 0, the sheet,
       weighted by the confidence): each vertex to the nearest local maximum >= --snap-min along its ray (the level set of the fitted
       W is smoother than the sheets and crosses the gaps between them) */
    uint8_t *sfa = nullptr; int sfn[3] = {0, 0, 0}; long long sfo[3] = {0, 0, 0}; double sfh = 1;
    const char *sfd = opt(argc, argv, "--snap-fields", nullptr);
    if (sfd) {
        if (snapz) { fprintf(stderr, "wraps: --snap or --snap-fields, not both\n"); return 2; }
        fit_fields_buf g; if (fit_fields_load(sfd, 0, 0, 0, &g)) { fprintf(stderr, "wraps: cannot read the fields in %s\n", sfd); return 1; }
        for (int d = 0; d < 3; d++) { sfo[d] = g.origin[d]; }
        sfh = g.h; fit_fields_free(&g);
        if (!(sfa = fit_field_array(sfd, "pc", 1, sfn))) { fprintf(stderr, "wraps: cannot read pc in %s\n", sfd); return 1; }
    }
    const double Z = m->D * m->h, Y = m->H * m->h, X = m->W * m->h;
    uint8_t *wf = nullptr; int wsh[3] = {0, 0, 0}; double wh = 0;   /* optional: the fields' data weight (max-pooled to <= ~1 GB) */
    const char *fd = opt(argc, argv, "--fields", nullptr);
    if (fd) {
        fit_fields_buf g; if (fit_fields_load(fd, 0, 0, 0, &g)) { fprintf(stderr, "wraps: cannot read the fields in %s\n", fd); return 1; }
        int pool = 1; while ((double)g.shape[0] * g.shape[1] * g.shape[2] / ((double)pool * pool * pool) > 1e9) pool *= 2;
        wh = (double)g.h * pool; fit_fields_free(&g);
        if (!(wf = fit_field_array(fd, "w", pool, wsh))) { fprintf(stderr, "wraps: cannot read the data weight in %s\n", fd); return 1; }
    }
    double wmin = 1e300, wmax = -1e300;   /* the wraps that meet the box */
    for (int a = 0; a <= 16; a++) for (int b = 0; b <= 16; b++) for (int c = 0; c <= 16; c++) {
        const double w = fit_set_W(fs, Z * a / 16, Y * b / 16, X * c / 16); if (w < wmin) wmin = w; if (w > wmax) wmax = w;
    }
    int k0 = (int)ceil(wmin), k1 = (int)floor(wmax);
    if (*opt(argc, argv, "--first", "")) { int f = atoi(opt(argc, argv, "--first", "")); if (f > k0) k0 = f; }
    if (*opt(argc, argv, "--last", "")) { int l = atoi(opt(argc, argv, "--last", "")); if (l < k1) k1 = l; }
    const int na = (int)Z + 1;
    double th0 = 1e300, th1 = -1e300; int full = 0;   /* the box's angular range around the axis */
    for (int zi = 0; zi <= 8; zi++) {
        const double z = Z * zi / 8, cy = m->cy[(int)fmin(z, na - 1)], cx = m->cx[(int)fmin(z, na - 1)];
        if (cy >= 0 && cy <= Y && cx >= 0 && cx <= X) full = 1;
        for (int c = 0; c < 4; c++) { const double t = atan2((c & 1 ? Y : 0) - cy, (c & 2 ? X : 0) - cx); if (t < th0) th0 = t; if (t > th1) th1 = t; }
    }
    if (!full && th1 - th0 > M_PI) {   /* the range straddles the -pi / pi cut: shift to [0, 2 pi) */
        th0 = 1e300; th1 = -1e300;
        for (int zi = 0; zi <= 8; zi++) {
            const double z = Z * zi / 8, cy = m->cy[(int)fmin(z, na - 1)], cx = m->cx[(int)fmin(z, na - 1)];
            for (int c = 0; c < 4; c++) { double t = atan2((c & 1 ? Y : 0) - cy, (c & 2 ? X : 0) - cx); if (t < 0) t += 2 * M_PI; if (t < th0) th0 = t; if (t > th1) th1 = t; }
        }
    }
    if (full) { th0 = -M_PI; th1 = M_PI; }
    const int rows = (int)floor(Z / step) + 1;
    /* the wraps on threads, each wrap's rows on one (a wrap's rows on all threads ran ~1.7 of 6 cores on a small box: per-wrap thread
       start-up, the serial tifxyz write); with snapping, one wrap at a time over all threads as before. Each wrap is computed as before. */
    wraps_all A = {fs, m, wf, {wsh[0], wsh[1], wsh[2]}, wh, step, Z, na, th0, th1, rows, k0, k1, nthreads, um, out_um, argv[3], snapz, snapo, snap_range, snap_min, sfa,
                   {sfn[0], sfn[1], sfn[2]}, {sfo[0], sfo[1], sfo[2]}, sfh, verbose_snap, Y, X, nullptr, 0, 0, 0, 0};
    wr_polar P = {0};
    if (wf && !getenv("UFSM_WRAPS_NOBOUND")) {   /* the rays' radius bounds (the table spans the box's angles, bins ~1e-3 rad) */
        P.rows = rows; P.t0 = th0 - 0.01; P.nb = (int)fmin(8192, fmax(64, ceil((th1 - th0 + 0.02) / 1e-3))); P.bw = (th1 - th0 + 0.02) / P.nb;
        P.rmin = malloc((size_t)rows * P.nb * sizeof(float)); P.rmax = malloc((size_t)rows * P.nb * sizeof(float));
        wr_polar_job PJ = {&P, m, wf, {wsh[0], wsh[1], wsh[2]}, wh, step, Z, na, 0};
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wr_polar_rows, &PJ);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        A.P = &P;
    }
    const int par = !(snapz || sfa) && nthreads > 1;
    if (par) {
        pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wraps_worker, &A);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    } else for (int k = k0; k <= k1; k++) if (wrap_one(&A, k, nthreads)) A.fail = 1;
    free(P.rmin); free(P.rmax);
    if (A.fail) return 1;
    const int nwr = A.nwr; const size_t nvalid_all = A.nvalid;
    fprintf(stderr, "wraps: %d wraps (%d..%d) with %zu vertices written to %s\n", nwr, k0, k1, nvalid_all, argv[3]);
    fit_set_free(fs); free(wf); free(sfa); if (snapz) z3_close(snapz);
    return 0;
}

/* ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8] [--tile 512] [--threads 16]
   Per wrap (each <wraps-dir>/wNNN tifxyz): the vertex grid bilinearly interpolated to scale pixels per native voxel, the
   surface normal from the grid derivatives, the volume (an ink zarr level, uint8) sampled trilinearly at t = -depth..depth native
   voxels along it and max-composited into <out-dir>/wNNN.tif (uint8); --ct also the CT at t = 0 into wNNN_ct.tif. Pixels whose
   cell has an invalid corner are 0. Rendered in output tiles of --tile pixels, each reading only the volume around its own
   vertices (+- depth), its pixel rows over --threads: any wrap size in bounded memory. */
static float trilin(const uint8_t *v, const int64_t n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return 0.f;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    if (z0 >= n[0] - 1) z0 = (int)n[0] - 2;
    if (y0 >= n[1] - 1) y0 = (int)n[1] - 2;
    if (x0 >= n[2] - 1) x0 = (int)n[2] - 2;
    z0 = z0 < 0 ? 0 : z0; y0 = y0 < 0 ? 0 : y0; x0 = x0 < 0 ? 0 : x0;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = z0 + (c >> 2 & 1), b = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        if (a >= n[0] || b >= n[1] || e >= n[2]) continue;
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[((size_t)a * n[1] + b) * n[2] + e];
    }
    return (float)s;
}
static float *read_tif_f32(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
/* ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase [--box z,y,x,nz,ny,nx] [--shard 512] [--cache DIR] [--threads 16]
   One uint8 array halved per axis (lossless volcomp) into <out-group>/<out-um> (an OME group of that one level): max (binary
   labels: any positive) or phase (the band phase codes, src/band.h: the circular mean of the known codes of the 2^3 block,
   PHASE_UNKNOWN where none is known or their resultant is under half their count). --box (source voxels) limits the work to
   the output shards it meets; others stay missing (the fill value). */
/* ufsm ct-hist <out.txt> <root> <key> z,y,x,n [z,y,x,n ...] [--cache DIR]
   The CT histogram of each box (n^3 at array key): one line of 256 counts per box (index 0 = masked air), boxes under 10% material
   skipped. The scanner augmentation's hist op (train / predict --aug-hist FILE --aug-hist-p P) maps a training window's material
   histogram onto one of these lines: boxes of the scans a model must generalise to (e.g. other scrolls). */
static int cmd_ct_hist(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm ct-hist <out.txt> <root> <key> z,y,x,n [z,y,x,n ...] [--cache DIR]\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "ct-hist: %s\n", z3_error()); return 1; }
    FILE *f = fopen(argv[2], "w"); if (!f) { fprintf(stderr, "ct-hist: cannot write %s\n", argv[2]); return 1; }
    int nw = 0;
    for (int a = 5; a < argc; a++) {
        if (argv[a][0] == '-') { a++; continue; }
        long long v[4]; if (sscanf(argv[a], "%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3) != 4 || v[3] < 1 || v[3] > 1024) { fprintf(stderr, "ct-hist: box %s: z,y,x,n (n <= 1024)\n", argv[a]); fclose(f); return 2; }
        const int64_t o[3] = {v[0], v[1], v[2]}, n[3] = {v[3], v[3], v[3]}; const size_t T = (size_t)v[3] * v[3] * v[3];
        uint8_t *b = malloc(T); if (!b) { fclose(f); return 1; }
        if (z3_read(z, o, n, b, 16)) { fprintf(stderr, "ct-hist: %s\n", z3_error()); free(b); fclose(f); return 1; }
        size_t h[256] = {0}; for (size_t k = 0; k < T; k++) h[b[k]]++;
        free(b);
        if (T - h[0] < T / 10) { fprintf(stderr, "ct-hist: box %s: %.1f%% material, skipped\n", argv[a], 100.0 * (T - h[0]) / T); continue; }
        for (int k = 0; k < 256; k++) fprintf(f, "%zu%c", h[k], k == 255 ? '\n' : ' ');
        nw++;
    }
    fclose(f); z3_close(z);
    fprintf(stderr, "ct-hist: %d histograms written to %s\n", nw, argv[2]);
    return nw ? 0 : 1;
}
/* cmd_downsample's pooling of one source slab (n[0] rows, even, from output row 0 of dst): output rows split over threads; phase
   mode with cos / sin tables (the same doubles as cos / sin of each code) */
static double g_ds_cs[PHASE_PERIOD], g_ds_sn[PHASE_PERIOD];
static void ds_tables(void) { for (int k = 0; k < PHASE_PERIOD; k++) { g_ds_cs[k] = cos(2 * M_PI * k / PHASE_PERIOD); g_ds_sn[k] = sin(2 * M_PI * k / PHASE_PERIOD); } }
static uint8_t g_ds_vr[256], g_ds_vv[256];   /* --mode votes: vote code -> recto / verso counts (src/votes.h) */
static void ds_vote_tables(void) { for (int k = 0; k < 256; k++) { int r, v; vote_split((uint8_t)k, &r, &v); g_ds_vr[k] = (uint8_t)r; g_ds_vv[k] = (uint8_t)v; } }
typedef struct { const uint8_t *src; int64_t n[3]; uint8_t *dst; int S; int64_t ny, nx, na; int phase; atomic_long next; } ds_job;   /* phase: 1 phase, 2 votes */
static void *ds_worker(void *arg) {
    ds_job *jb = arg; const int64_t *n = jb->n; const int S = jb->S;
    for (int64_t a; (a = atomic_fetch_add(&jb->next, 1)) < jb->na;) {
        const uint8_t *r0 = jb->src + (size_t)(2 * a) * n[1] * n[2], *r1 = 2 * a + 1 < n[0] ? r0 + (size_t)n[1] * n[2] : nullptr;
        for (int64_t b = 0; b < jb->ny; b++) {
            uint8_t *out = jb->dst + ((size_t)a * S + b) * S;
            const int64_t y0 = 2 * b, y1 = 2 * b + 1 < n[1] ? 2 * b + 1 : -1;
            for (int64_t c = 0; c < jb->nx; c++) {
                const int64_t x0 = 2 * c, x1 = 2 * c + 1 < n[2] ? 2 * c + 1 : -1;
                int v[8], nv = 0;
                for (int k = 0; k < 8; k++) {   /* the old order: z, y, x bits */
                    const uint8_t *r = k >> 2 & 1 ? r1 : r0; const int64_t yy = k >> 1 & 1 ? y1 : y0, xx = k & 1 ? x1 : x0;
                    if (r && yy >= 0 && xx >= 0) v[nv++] = r[(size_t)yy * n[2] + xx];
                }
                if (!jb->phase) { int mx = 0; for (int k = 0; k < nv; k++) mx = v[k] > mx ? v[k] : mx; out[c] = (uint8_t)mx; continue; }
                if (jb->phase == 2) { int r = 0, vv = 0; for (int k = 0; k < nv; k++) { r += g_ds_vr[v[k]]; vv += g_ds_vv[v[k]]; } out[c] = vote_code(r, vv); continue; }
                double cs = 0, sn = 0; int known = 0;
                for (int k = 0; k < nv; k++) if (v[k] < PHASE_PERIOD) { cs += g_ds_cs[v[k]]; sn += g_ds_sn[v[k]]; known++; }
                uint8_t o = PHASE_UNKNOWN;
                if (known && hypot(cs, sn) >= 0.5 * known) { double t = atan2(sn, cs) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)((int)lround(t * PHASE_PERIOD) % PHASE_PERIOD); }
                out[c] = o;
            }
        }
    }
    return nullptr;
}
static void ds_run(ds_job *jb, int nt) {
    atomic_store(&jb->next, 0);
    nt = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    pthread_t th[64]; int k = 0;
    for (; k < nt - 1 && pthread_create(&th[k], nullptr, ds_worker, jb) == 0; k++) {}
    ds_worker(jb);
    for (int i = 0; i < k; i++) pthread_join(th[i], nullptr);
}
static int cmd_downsample(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase|votes [--box z,y,x,nz,ny,nx] [--shard 512] [--cache DIR] [--threads 16]\n"); return 2; }
    const char *mode = opt(argc, argv, "--mode", "max"); const int votes = !strcmp(mode, "votes"), phase = !strcmp(mode, "phase") ? 1 : votes ? 2 : 0;
    if (!phase && strcmp(mode, "max")) { fprintf(stderr, "downsample: --mode max|phase|votes\n"); return 2; }
    if (votes) ds_vote_tables();
    const int S = atoi(opt(argc, argv, "--shard", "512")), nt = atoi(opt(argc, argv, "--threads", "16")); const double um = atof(argv[5]);
    z3 *z = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int fill = phase == 1 ? PHASE_UNKNOWN : m->fill;   /* votes: 0 = no votes */
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    if (*opt(argc, argv, "--box", "")) { long long v[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "downsample: --box z,y,x,nz,ny,nx\n"); return 2; } for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } }
    const int64_t osh[3] = {(m->shape[0] + 1) / 2, (m->shape[1] + 1) / 2, (m->shape[2] + 1) / 2};
    char lv[32], od[1400], at[400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"%s halved (%s)\"%s}}", argv[3], mode, m->label_binary ? ",\"encoding\":\"binary\"" : votes ? ",\"encoding\":\"side-votes\",\"code\":\"src/votes.h: the votes of each 2^3 block added\"" : "");
    z3w *w = z3w_create(od, osh, S, 0, fill, at);
    double lum[16]; int nl = 0;   /* the group lists every level already in it (each a number-named array), sorted, plus this one */
    { DIR *gd = opendir(argv[4]); struct dirent *ge;
      while (gd && (ge = readdir(gd)) && nl < 15) { char *e; const double v = strtod(ge->d_name, &e); char zj[1400]; snprintf(zj, sizeof zj, "%s/%s/zarr.json", argv[4], ge->d_name);
          if (v > 0 && !*e && fabs(v - um) > 1e-6 && access(zj, R_OK) == 0) lum[nl++] = v; }
      if (gd) closedir(gd); }
    lum[nl++] = um;
    for (int a = 1; a < nl; a++) for (int b = a; b > 0 && lum[b] < lum[b - 1]; b--) { const double t = lum[b]; lum[b] = lum[b - 1]; lum[b - 1] = t; }
    if (!w || z3w_write_group(argv[4], lum, nl, "ufsm-downsample", nullptr)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
    /* per output shard: its 2S source cube read in slabs of the source chunk depth (each inner chunk decoded once; reading two planes
       at a time decoded every chunk it met once per plane pair), the slab pooled on nt threads; output shards whose source shards are
       all absent are skipped (all fill: z3w writes nothing for them either) */
    int64_t sd = m->chunk[0] > 0 ? m->chunk[0] : 128; if (sd % 2) sd *= 2; if (sd > 2 * (int64_t)S) sd = 2 * (int64_t)S;   /* source planes per slab */
    uint8_t *sb = malloc((size_t)S * S * S), *src = malloc((size_t)sd * 2 * S * 2 * S);
    if (!sb || !src) { fprintf(stderr, "downsample: out of memory\n"); return 1; }
    ds_tables();
    int nsh = 0, nskip = 0;
    for (int64_t sz = (bo[0] / 2) / S; sz * S < osh[0] && sz * S * 2 < bo[0] + bn[0]; sz++)
    for (int64_t sy = (bo[1] / 2) / S; sy * S < osh[1] && sy * S * 2 < bo[1] + bn[1]; sy++)
    for (int64_t sx = (bo[2] / 2) / S; sx * S < osh[2] && sx * S * 2 < bo[2] + bn[2]; sx++) {
        const int64_t so[3] = {sz, sy, sx}; int present = 0;   /* the source shards under this output shard */
        int64_t q0[3], q1[3];
        for (int d = 0; d < 3; d++) {
            const int64_t a = 2 * so[d] * S, e = 2 * (so[d] + 1) * S < m->shape[d] ? 2 * (so[d] + 1) * S : m->shape[d];
            q0[d] = a / m->shard[d]; q1[d] = (e - 1) / m->shard[d];
        }
        for (int64_t a = q0[0]; a <= q1[0] && !present; a++) for (int64_t b = q0[1]; b <= q1[1] && !present; b++) for (int64_t c = q0[2]; c <= q1[2] && !present; c++)
            present = z3_shard_present(z, a, b, c) != 0;   /* (-1, an error: read it and let the read report) */
        if (!present) { nskip++; continue; }
        memset(sb, fill, (size_t)S * S * S);
        const int64_t ny = osh[1] - sy * S < S ? osh[1] - sy * S : S, nx = osh[2] - sx * S < S ? osh[2] - sx * S : S;
        for (int64_t a0 = 0; a0 < S && sz * S + a0 < osh[0]; a0 += sd / 2) {   /* output rows a0 .. from source rows 2 a0 .. */
            const int64_t o[3] = {2 * (sz * S + a0), 2 * sy * S, 2 * sx * S};
            const int64_t n[3] = {m->shape[0] - o[0] < sd ? m->shape[0] - o[0] : sd, m->shape[1] - o[1] < 2 * ny ? m->shape[1] - o[1] : 2 * ny, m->shape[2] - o[2] < 2 * nx ? m->shape[2] - o[2] : 2 * nx};
            if (z3_read(z, o, n, src, nt)) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
            const size_t sn_ = (size_t)n[0] * n[1] * n[2]; size_t k = 0;
            while (k < sn_ && src[k] == (uint8_t)m->fill) k++;
            if (k == sn_ && fill == m->fill) continue;   /* all fill in, all fill out (max of fills; phase: unknown) */
            int64_t na = (n[0] + 1) / 2; if (a0 + na > S) na = S - a0; if (sz * S + a0 + na > osh[0]) na = osh[0] - sz * S - a0;
            ds_job jb = {src, {n[0], n[1], n[2]}, sb + (size_t)a0 * S * S, S, ny, nx, na, phase, 0};
            ds_run(&jb, nt);
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, nt)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
        nsh++;
    }
    if (nskip) fprintf(stderr, "downsample: %d output shards skipped (no source shard)\n", nskip);
    z3w_close(w); free(sb); free(src); z3_close(z);
    fprintf(stderr, "downsample: %d shards of %s written to %s\n", nsh, lv, od);
    return 0;
}

/* ufsm side-votes <out-group> <out-um> <root> <key> [<root> <key> ...] [--shard 512] [--threads 16] [--cache DIR]
   Recto / verso side labels of several surface sets of one scroll (each phase-raster --pair 2 on one set's codes, or an earlier
   side-votes array) added into one vote array (src/votes.h: per voxel the sets calling it recto side and verso side, 15 at most),
   lossless volcomp in <out-group>/<out-um> (an OME group of that level; downsample --mode votes for the coarser ones). All inputs on
   one grid (equal shapes). Sources read it with "encoding": "side-votes" (unanimous / majority / soft).
   ufsm codes-merge (same arguments): winding_mod14 code rasters (the published band codes, raster --value of wind-assign's q.tif)
   united instead: codes within 4 steps are the same wrap (the first kept), else 255; phase-raster --pair 2 then labels between
   wraps from any of them. */
static int cmd_side_votes(int argc, char **argv) {
    const int codes = !strcmp(argv[1], "codes-merge");
    if (argc < 6) { fprintf(stderr, "usage: ufsm %s <out-group> <out-um> <root> <key> [<root> <key> ...] [--shard 512] [--threads 16] [--cache DIR]\n", argv[1]); return 2; }
    const int S = atoi(opt(argc, argv, "--shard", "512")), nt = atoi(opt(argc, argv, "--threads", "16")); const double um = atof(argv[3]);
    if (S < 128 || S % 128 || !(um > 0)) { fprintf(stderr, "side-votes: out-um > 0, --shard a multiple of 128\n"); return 2; }
    z3 *in[64]; const char *names[64]; char nb[64][1200]; int nin = 0;
    for (int a = 4; a + 1 < argc && nin < 64; a += 2) {
        if (argv[a][0] == '-') break;
        in[nin] = z3_open(store_open(argv[a]), argv[a + 1], opt(argc, argv, "--cache", nullptr));
        if (!in[nin]) { fprintf(stderr, "side-votes: %s %s: %s\n", argv[a], argv[a + 1], z3_error()); return 1; }
        snprintf(nb[nin], sizeof nb[nin], "%s/%s", argv[a], argv[a + 1]); names[nin] = nb[nin]; nin++;
    }
    size_t st[2] = {0, 0}; int nsh = 0;
    const int rc = (codes ? codes_merge : side_votes_merge)(argv[2], um, in, names, nin, S, nt, &nsh, st);
    for (int i = 0; i < nin; i++) z3_close(in[i]);
    if (rc) { fprintf(stderr, "%s: %s\n", argv[1], side_votes_error()); return 1; }
    if (codes) fprintf(stderr, "codes-merge: %d inputs, %d shards -> %s; %zu coded voxels, %.2f%% of them conflicts (255)\n", nin, nsh, argv[2], st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0);
    else fprintf(stderr, "side-votes: %d inputs, %d shards -> %s; %zu voxels with votes, %.2f%% of them disputed (recto and verso votes)\n", nin, nsh, argv[2], st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0);
    return 0;
}

/* ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>... [--ref-um 4.8] [--vol-um 2.4] [--kind codes|winding] [--tol 6]
   [--step 4] [--dir 0] [--axis JSON] [--axis-um U] [--drop-duplicates 0]
   Winding ids for surfaces of other authors / runs (src/wind.h): each tifxyz gets q = K + dir theta / 2 pi per vertex in the gauge of
   a reference winding raster (codes: raster --value / the published band codes; winding: phase-raster --winding 1, dense), written
   as <out-dir>/<name> (x / y / z copied, q.tif float32, NaN where dropped or unassigned) for `ufsm raster --value q.tif`; one line
   per surface on stdout and in <out-dir>/assign.jsonl. --tol: label voxels from the reference that count as on it; --vol-um: the
   surfaces' scan voxel; --axis-um: the axis JSON's voxel (default --vol-um); --out-um: the voxel of the written coordinates (default
   --vol-um; the reference's --ref-um puts them on its grid for raster --value and codes-merge). */
typedef struct { z3 *z; int kind; } wa_ref;
typedef struct { int64_t key; size_t i; } wa_pt;
static int cmp_wa_pt(const void *a, const void *b) { const int64_t x = ((const wa_pt *)a)->key, y = ((const wa_pt *)b)->key; return x < y ? -1 : x > y; }
static void wa_lookup(void *ud, const float *pts, size_t n, float tol, float *dist, float *qref) {   /* per 128^3 block of points: one haloed read */
    const wa_ref *R = ud; const int H = (int)ceilf(tol), B = 128, S = B + 2 * H; const z3_meta *m = z3_meta_of(R->z);
    wa_pt *ord = malloc(n * sizeof *ord); uint8_t *blk = malloc((size_t)S * S * S);
    if (!ord || !blk) { free(ord); free(blk); for (size_t i = 0; i < n; i++) { dist[i] = INFINITY; qref[i] = NAN; } return; }
    for (size_t i = 0; i < n; i++) {
        const int64_t bz = (int64_t)floorf(pts[3 * i] / B), by = (int64_t)floorf(pts[3 * i + 1] / B), bx = (int64_t)floorf(pts[3 * i + 2] / B);
        ord[i] = (wa_pt){((bz + 4096) << 40) | ((by + 4096) << 20) | (bx + 4096), i};
    }
    qsort(ord, n, sizeof *ord, cmp_wa_pt);
    for (size_t a = 0; a < n;) {
        size_t b = a; while (b < n && ord[b].key == ord[a].key) b++;
        const size_t i0 = ord[a].i;
        const int64_t o[3] = {(int64_t)floorf(pts[3 * i0] / B) * B - H, (int64_t)floorf(pts[3 * i0 + 1] / B) * B - H, (int64_t)floorf(pts[3 * i0 + 2] / B) * B - H}, nn[3] = {S, S, S};
        int inside = 1; for (int d = 0; d < 3; d++) inside &= o[d] + S > 0 && o[d] < m->shape[d];
        const int ok = inside && !z3_read(R->z, o, nn, blk, 1);
        for (size_t t = a; t < b; t++) {
            const size_t i = ord[t].i; dist[i] = INFINITY; qref[i] = NAN;
            if (!ok) continue;
            const int z0 = (int)lroundf(pts[3 * i]) - (int)o[0], y0 = (int)lroundf(pts[3 * i + 1]) - (int)o[1], x0 = (int)lroundf(pts[3 * i + 2]) - (int)o[2];
            for (int z = z0 - H; z <= z0 + H; z++) for (int y = y0 - H; y <= y0 + H; y++) for (int x = x0 - H; x <= x0 + H; x++) {
                if (z < 0 || z >= S || y < 0 || y >= S || x < 0 || x >= S) continue;
                const float q = wind_code_q(blk[((size_t)z * S + y) * S + x], R->kind);
                if (isnan(q)) continue;
                const float dz = z + o[0] - pts[3 * i], dy = y + o[1] - pts[3 * i + 1], dx = x + o[2] - pts[3 * i + 2], d = sqrtf(dz * dz + dy * dy + dx * dx);
                if (d < dist[i]) { dist[i] = d; qref[i] = q; }
            }
        }
        a = b;
    }
    free(ord); free(blk);
}
typedef struct { axis ax; double f; } wa_axis;
static void wa_axis_fn(void *ud, double z, double *cy, double *cx) { const wa_axis *A = ud; axis_at(&A->ax, z / A->f, cy, cx); *cy *= A->f; *cx *= A->f; }
static float *wa_read_f32(const char *dir, const char *name, int *w, int *h) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name);
    tiff *t = tiff_open_file(p); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 || (*w && (pg.w != *w || pg.h != *h))) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
static int cmd_wind_assign(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>... [--ref-um 4.8] [--vol-um 2.4] [--kind codes|winding] [--tol 6] [--step 4] [--dir 0] [--axis JSON] [--axis-um U] [--drop-duplicates 0]\n"); return 2; }
    const double ref_um = atof(opt(argc, argv, "--ref-um", "4.8")), vol_um = atof(opt(argc, argv, "--vol-um", "2.4")), axis_um = atof(opt(argc, argv, "--axis-um", opt(argc, argv, "--vol-um", "2.4")));
    const char *kind = opt(argc, argv, "--kind", "codes");
    wind_params p = {vol_um / ref_um, (float)atof(opt(argc, argv, "--tol", "6")), atoi(opt(argc, argv, "--step", "4")), atoi(opt(argc, argv, "--dir", "0")),
                     atoi(opt(argc, argv, "--drop-duplicates", "0")), !strcmp(kind, "winding")};
    if (!(ref_um > 0 && vol_um > 0 && axis_um > 0) || (strcmp(kind, "codes") && strcmp(kind, "winding")) || !(p.tol >= 0 && p.tol <= 32) || p.step < 1 || (p.dir != 0 && p.dir != 1 && p.dir != -1)) {
        fprintf(stderr, "wind-assign: --ref-um / --vol-um / --axis-um > 0, --kind codes|winding, --tol 0..32, --step >= 1, --dir -1|0|1\n"); return 2;
    }
    wa_ref R = {z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)), p.dense ? WIND_WINDING : WIND_CODES};
    if (!R.z) { fprintf(stderr, "wind-assign: %s\n", z3_error()); return 1; }
    wa_axis A = {.f = axis_um / vol_um};
    if (axis_load(&A.ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "wind-assign: cannot load the axis\n"); return 1; }
    char lp[1400]; snprintf(lp, sizeof lp, "%s/assign.jsonl", argv[4]);
    if (mkdir(argv[4], 0755) && errno != EEXIST) { fprintf(stderr, "wind-assign: mkdir %s\n", argv[4]); return 1; }
    FILE *log = fopen(lp, "a"); if (!log) { fprintf(stderr, "wind-assign: cannot write %s\n", lp); return 1; }
    int nsurf = 0, rc = 0;
    for (int a = 5; a < argc; a++) {
        if (argv[a][0] == '-') { a++; continue; }
        int w = 0, h = 0; float *X = wa_read_f32(argv[a], "x.tif", &w, &h), *Y = X ? wa_read_f32(argv[a], "y.tif", &w, &h) : nullptr, *Z = Y ? wa_read_f32(argv[a], "z.tif", &w, &h) : nullptr;
        if (!Z) { fprintf(stderr, "wind-assign: %s: x / y / z.tif must be matching float32 images\n", argv[a]); free(X); free(Y); rc = 1; continue; }
        const size_t N = (size_t)w * h; float *xyz = malloc(N * 3 * sizeof *xyz); uint8_t *valid = malloc(N);
        for (size_t i = 0; i < N; i++) {
            xyz[3 * i] = X[i]; xyz[3 * i + 1] = Y[i]; xyz[3 * i + 2] = Z[i];
            valid[i] = isfinite(X[i]) && isfinite(Y[i]) && isfinite(Z[i]) && Z[i] > 0 && X[i] >= 0;   /* (load_mesh's rule; -1 = invalid) */
        }
        wind_result r;
        if (wind_assign(xyz, valid, w, h, wa_axis_fn, &A, wa_lookup, &R, p, &r)) { fprintf(stderr, "wind-assign: out of memory\n"); return 1; }
        const char *base = strrchr(argv[a], '/'); base = base && base[1] ? base + 1 : argv[a];
        char od[1400], qp[1500]; snprintf(od, sizeof od, "%s/%s", argv[4], base);
        const double out_um = atof(opt(argc, argv, "--out-um", opt(argc, argv, "--vol-um", "2.4"))), f = vol_um / (out_um > 0 ? out_um : vol_um);   /* --out-um: coordinates in that voxel (e.g. the reference grid's, for raster --value) */
        if (f != 1.0) for (size_t i = 0; i < N; i++) if (valid[i]) { X[i] = (float)(X[i] * f); Y[i] = (float)(Y[i] * f); Z[i] = (float)(Z[i] * f); }
        if (tifxyz_write(od, w, h, X, Y, Z, 1.0, out_um > 0 ? out_um : vol_um, -1, "ufsm wind-assign (x / y / z of the input; q.tif: the winding in the reference's gauge)")) { fprintf(stderr, "wind-assign: cannot write %s\n", od); rc = 1; }
        snprintf(qp, sizeof qp, "%s/q.tif", od);
        if (tiffw_write(qp, w, h, 3, r.q, 1)) { fprintf(stderr, "wind-assign: cannot write %s\n", qp); rc = 1; }
        size_t nq = 0, nf[5] = {0}; for (size_t i = 0; i < N; i++) { nq += !isnan(r.q[i]); nf[r.flag[i]]++; }
        for (int c = 0; c < r.ncomp; c++) {
            const wind_comp *k = &r.comp[c];
            printf("%s comp %d (%zu vertices): %s dir %+d K %d (mod 14), %zu of %zu samples on the reference, %zu agree, median dist %.2f, |resid| %.3f turn\n",
                   base, c, k->nvalid, wind_verdict_name(k->verdict), k->dir, k->K, k->nmatch, k->nsample, k->nagree, k->median_dist, k->mean_abs_resid);
            fprintf(log, "{\"surface\":\"%s\",\"comp\":%d,\"vertices\":%zu,\"verdict\":\"%s\",\"dir\":%d,\"K\":%d,\"samples\":%zu,\"matched\":%zu,\"agree\":%zu,\"median_dist\":%.3f,\"resid\":%.4f,\"q_vertices\":%zu,\"dropped_conflict\":%zu,\"dropped_duplicate\":%zu,\"unassigned\":%zu}\n",
                    base, c, k->nvalid, wind_verdict_name(k->verdict), k->dir, k->K, k->nsample, k->nmatch, k->nagree, isfinite(k->median_dist) ? k->median_dist : -1.0, k->mean_abs_resid, nq, nf[2], nf[3], nf[4]);
        }
        wind_result_free(&r); free(xyz); free(valid); free(X); free(Y); free(Z); nsurf++;
    }
    fclose(log); z3_close(R.z);
    fprintf(stderr, "wind-assign: %d surfaces -> %s (then: ufsm raster OUT --value q.tif --binary 0 --levels 1 ... %s/*)\n", nsurf, argv[4], argv[4]);
    return rc;
}

/* ufsm wrapdist <wraps-dir> <labels-root> <labels-key> [--max 32] [--chunk 128] [--threads 16] [--per-wrap]
   How far the wraps' vertices (every <wraps-dir>/wNNN tifxyz) lie from the nearest labelled sheet voxel (label value != 0) of
   a label array whose key is its voxel size in um (e.g. 4.8; the wraps' voxel_size_um from meta.json scales the vertices onto
   it). Exact distance transform (src/edt.c) over the vertices' bounding box in z-chunks of --chunk label rows with --max rows
   of overlap: distances up to --max label voxels are exact, larger ones reported as beyond it. Memory ~5 bytes per voxel of
   one chunk. */
static float *wd_read(const char *path, int *w, int *h);
static int cmd_wrapdist(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wrapdist <wraps-dir> <labels-root> <labels-key> [--max 32] [--chunk 128] [--threads 16] [--per-wrap] [--cache DIR]\n"); return 2; }
    const int M = atoi(opt(argc, argv, "--max", "32")), C = atoi(opt(argc, argv, "--chunk", "128")), nt = atoi(opt(argc, argv, "--threads", "16"));
    const int per = arg_index(argc, argv, "--per-wrap") > 0; const double lum = atof(argv[4]);
    if (!(lum > 0) || M < 1 || C < 1) { fprintf(stderr, "wrapdist: the labels key must be its voxel size in um (e.g. 4.8); --max, --chunk >= 1\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!z) { fprintf(stderr, "wrapdist: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(z);
    /* vertices of every wrap, in label voxels */
    size_t nv = 0, cap = 1 << 20; float (*v)[3] = malloc(cap * sizeof *v); int *wid = malloc(cap * sizeof(int)); int nw = 0, ids[4096];
    DIR *dd = opendir(argv[2]); struct dirent *de; if (!dd) { fprintf(stderr, "wrapdist: cannot open %s\n", argv[2]); return 1; }
    while ((de = readdir(dd)) && nw < 4096) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = wd_read(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = wd_read(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = wd_read(p, &w3, &h3);
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name); double wum = 0;
        { FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); const char *q = strstr(b, "\"voxel_size_um\":"); if (q) wum = atof(q + 16); } }
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(wum > 0)) { free(X); free(Y); free(Z); continue; }
        const double sc = wum / lum;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            if (Z[i] < 0) continue;
            if (nv == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); wid = realloc(wid, cap * sizeof(int)); }
            v[nv][0] = (float)(Z[i] * sc); v[nv][1] = (float)(Y[i] * sc); v[nv][2] = (float)(X[i] * sc); wid[nv++] = nw;
        }
        ids[nw++] = atoi(de->d_name + 1); free(X); free(Y); free(Z);
    }
    closedir(dd);
    if (!nv) { fprintf(stderr, "wrapdist: no vertices in %s\n", argv[2]); return 1; }
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < nv; i++) for (int d = 0; d < 3; d++) { if (v[i][d] < lo[d]) lo[d] = v[i][d]; if (v[i][d] > hi[d]) hi[d] = v[i][d]; }
    int64_t bo[3], bn[3];
    for (int d = 0; d < 3; d++) { bo[d] = (int64_t)floor(lo[d]) - M; if (bo[d] < 0) bo[d] = 0; int64_t e = (int64_t)ceil(hi[d]) + M + 1; if (e > zm->shape[d]) e = zm->shape[d]; bn[d] = e - bo[d]; }
    float *dist = malloc(nv * sizeof(float)); for (size_t i = 0; i < nv; i++) dist[i] = INFINITY;
    const size_t plane = (size_t)bn[1] * bn[2];
    uint8_t *lab = malloc((size_t)(C + 2 * M) * plane); float *dsq = malloc((size_t)(C + 2 * M) * plane * sizeof(float));
    if (!lab || !dsq) { fprintf(stderr, "wrapdist: no memory for a chunk of %lld rows of %zu voxels (lower --chunk)\n", (long long)(C + 2 * M), plane); return 1; }
    for (int64_t z0 = bo[0]; z0 < bo[0] + bn[0]; z0 += C) {
        const int64_t a = z0 - M < bo[0] ? bo[0] : z0 - M, b = z0 + C + M > bo[0] + bn[0] ? bo[0] + bn[0] : z0 + C + M;
        int any = 0; for (size_t i = 0; i < nv && !any; i++) any = v[i][0] >= z0 - 0.5 && v[i][0] < z0 + C - 0.5;
        if (!any) continue;
        const int64_t o[3] = {a, bo[1], bo[2]}, n[3] = {b - a, bn[1], bn[2]};
        if (z3_read(z, o, n, lab, nt)) { fprintf(stderr, "wrapdist: %s\n", z3_error()); return 1; }
        if (edt_sq(lab, n, dsq, nt)) { fprintf(stderr, "wrapdist: out of memory\n"); return 1; }
        for (size_t i = 0; i < nv; i++) {
            const int64_t q0 = (int64_t)lround(v[i][0]); if (q0 < z0 || q0 >= z0 + C) continue;
            const int64_t q1 = (int64_t)lround(v[i][1]) - bo[1], q2 = (int64_t)lround(v[i][2]) - bo[2];
            if (q1 < 0 || q2 < 0 || q1 >= bn[1] || q2 >= bn[2]) continue;
            const float s = dsq[((size_t)(q0 - a) * bn[1] + q1) * bn[2] + q2];
            dist[i] = s >= EDT_INF ? (float)(M + 1) : fminf(sqrtf(s), (float)(M + 1));
        }
    }
    free(lab); free(dsq); z3_close(z);
    /* stats in label voxels and um; per wrap optionally */
    for (int w = -1; w < (per ? nw : 0); w++) {   /* all, then each wrap */
        size_t k = 0; float *a = malloc(nv * sizeof(float));
        for (size_t i = 0; i < nv; i++) if (isfinite(dist[i]) && (w < 0 || wid[i] == w)) a[k++] = dist[i];
        if (k) {
            qsort(a, k, sizeof(float), cmp_fl);
            size_t c1 = 0, c2 = 0, c4 = 0, cm = 0; for (size_t i = 0; i < k; i++) { c1 += a[i] * lum <= 9.6; c2 += a[i] * lum <= 19.2; c4 += a[i] * lum > 38.4; cm += a[i] > M; }
            if (w < 0) printf("wrapdist: %zu vertices of %d wraps (%zu outside the labels' extent); distance to the nearest labelled voxel (label voxel %.3g um):\n", k, nw, nv - k, lum);
            char tag[16]; if (w < 0) snprintf(tag, sizeof tag, "all"); else snprintf(tag, sizeof tag, "w%03d", ids[w]);
            printf("  %-5s %8zu vertices: median %.1f um, p75 %.1f, p90 %.1f; <= 9.6 um %.3f, <= 19.2 um %.3f, > 38.4 um %.3f, beyond --max (%.0f um) %.3f\n",
                   tag, k, a[k / 2] * lum, a[k * 3 / 4] * lum, a[k * 9 / 10] * lum, (double)c1 / k, (double)c2 / k, (double)c4 / k, M * lum, (double)cm / k);
        }
        free(a);
    }
    free(v); free(wid); free(dist);
    return 0;
}
static float *wd_read(const char *path, int *w, int *h) { return read_tif_f32(path, w, h); }

/* ufsm wrapswitch <wraps-dir> <codes-root> <codes-key> [--range 30] [--gap 1] [--min-run 100] [--chunk 64] [--threads 16] [--per-wrap] [--cache DIR]
   How often the wraps (every <wraps-dir>/wNNN tifxyz) switch sheets. Each vertex takes the code of the nearest labelled voxel along
   its surface normal within +- --range um (band codes: 1 + round(18 q) mod 252, q the published wrap's unwrapped turn coordinate:
   continuous along a sheet, one turn apart between neighbouring sheets). Walking each grid row (around the axis) and each column
   (along z), consecutive coded vertices less than --gap mm apart whose q differ by more than half a turn (mod 14) are a switch to
   another sheet, once the new sheet holds for --min-run um (shorter runs: a vertex between two sheets flickering between their
   labels). Being a whole turn off everywhere costs nothing here: only changes count. Reports switches per cm of coded path
   along theta and z, the coded share of the vertices and the mean switch-free run, over all wraps (and per wrap). The codes are read
   in z chunks of --chunk label rows over the vertices' bounding box. */
typedef struct { int w, h, id; float *X, *Y, *Z; uint8_t *code; float *dist; } ws_wrap;   /* dist: um along the normal to the code */
typedef struct {
    ws_wrap *W; int nw; long *rs; long nrows; double sc, R; const uint8_t *lab; int64_t o[3], n[3], z0, z1; _Atomic long next; double lum;
    const axis *ax; double s0;   /* with --axis: the normals turned away from the axis, dist signed (+: the code lies outward) */
} ws_job;
static void *ws_rows(void *arg) {
    ws_job *J = arg;
    for (long g; (g = atomic_fetch_add(&J->next, 1)) < J->nrows;) {
        int lo = 0, hi = J->nw - 1; while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (J->rs[mid] <= g) lo = mid; else hi = mid - 1; }
        ws_wrap *w = &J->W[lo]; const int r = (int)(g - J->rs[lo]), W = w->w, H = w->h;
        for (int c = 0; c < W; c++) {
            const size_t i = (size_t)r * W + c; if (w->Z[i] < 0 || w->code[i]) continue;
            const double P[3] = {w->Z[i] * J->sc, w->Y[i] * J->sc, w->X[i] * J->sc};
            const int64_t q0 = (int64_t)llround(P[0]); if (q0 < J->z0 || q0 >= J->z1) continue;   /* this chunk's vertices */
            const int cl = c > 0 && w->Z[i - 1] >= 0 ? c - 1 : c, cr = c + 1 < W && w->Z[i + 1] >= 0 ? c + 1 : c;
            const int ru = r > 0 && w->Z[i - W] >= 0 ? r - 1 : r, rd = r + 1 < H && w->Z[i + W] >= 0 ? r + 1 : r;
            if (cl == cr || ru == rd) continue;
            const size_t a = (size_t)r * W + cl, b = (size_t)r * W + cr, u = (size_t)ru * W + c, d = (size_t)rd * W + c;
            const double du[3] = {w->Z[b] - w->Z[a], w->Y[b] - w->Y[a], w->X[b] - w->X[a]}, dv[3] = {w->Z[d] - w->Z[u], w->Y[d] - w->Y[u], w->X[d] - w->X[u]};
            double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
            const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (!(nl > 0)) continue;
            for (int k = 0; k < 3; k++) N[k] /= nl;
            if (J->ax) {   /* outward: away from the axis (the published codes' winding grows outward) */
                double cy, cx; axis_at(J->ax, w->Z[i] * J->s0, &cy, &cx);
                if (N[1] * (w->Y[i] * J->s0 - cy) + N[2] * (w->X[i] * J->s0 - cx) < 0) for (int k = 0; k < 3; k++) N[k] = -N[k];
            }
            for (int s = 0; s <= 4 * (int)ceil(J->R); s++) {   /* t = 0, +.5, -.5, +1, -1, ...: the nearest labelled voxel first */
                const double t = (s + 1) / 2 * 0.5 * (s % 2 ? 1 : -1); if (fabs(t) > J->R) break;
                int64_t p[3]; int in = 1;
                for (int k = 0; k < 3; k++) { p[k] = (int64_t)llround(P[k] + t * N[k]) - J->o[k]; in &= p[k] >= 0 && p[k] < J->n[k]; }
                if (!in) continue;
                const uint8_t v = J->lab[((size_t)p[0] * J->n[1] + p[1]) * J->n[2] + p[2]];
                if (v >= 1 && v <= 252) { w->code[i] = v; w->dist[i] = (float)((J->ax ? t : fabs(t)) * J->lum); break; }
            }
        }
    }
    return nullptr;
}
typedef struct { double path, sw, runs; } ws_stat;
/* one direction over the grid: per line, coded vertices joined into paths (consecutive ones less than gap_um apart); along a path
   the sheet index unwraps the codes (s += round(dq), dq the code change mod 14 turns); runs of one sheet shorter than min_um
   (a vertex between two sheets flickering between their labels) are dropped, and a switch is a change between the remaining runs */
static void ws_walk(const ws_wrap *w, int along_z, double um, double gap_um, double min_um, ws_stat *s, int *sb, double *lb) {
    const int L = along_z ? w->w : w->h, M = along_z ? w->h : w->w;
    for (int l = 0; l < L; l++) {
        int n = 0, last = -1;
        for (int m = 0; m <= M; m++) {
            const size_t i = m < M ? (along_z ? (size_t)m * w->w + l : (size_t)l * w->w + m) : 0;
            const int coded = m < M && w->code[i];
            double dl = 0;
            if (coded && last >= 0) {
                const size_t j = along_z ? (size_t)last * w->w + l : (size_t)l * w->w + last;
                const double dz = w->Z[i] - w->Z[j], dy = w->Y[i] - w->Y[j], dx = w->X[i] - w->X[j]; dl = sqrt(dz * dz + dy * dy + dx * dx) * um;
            }
            if (n > 0 && (!coded ? m == M : dl > gap_um)) {   /* the path ends: its runs */
                double run = 0; int cur = 0, have = 0; double path = 0;
                for (int k = 0; k < n;) {
                    int e = k; double len = 0; while (e < n && sb[e] == sb[k]) { len += lb[e]; e++; }
                    path += len;
                    if (len >= min_um || (k == 0 && e == n)) {
                        if (have && sb[k] != cur) { s->sw += 1; s->runs += 1; run = 0; }
                        cur = sb[k]; have = 1; run += len;
                    } else run += len;
                    k = e;
                }
                s->path += path; if (run > 0) s->runs += 1;
                n = 0;
            }
            if (!coded) continue;
            if (n == 0) { sb[0] = 0; lb[0] = 0; n = 1; }
            else { const size_t j = along_z ? (size_t)last * w->w + l : (size_t)l * w->w + last;
                   double dq = (w->code[i] - w->code[j]) / 18.0; dq = fmod(dq + 7 + 14, 14) - 7;
                   sb[n] = sb[n - 1] + (int)lround(dq); lb[n] = dl; n++; }
            last = m;
        }
    }
}
static int cmd_wrapswitch(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wrapswitch <wraps-dir> <codes-root> <codes-key> [--range 30] [--gap 1] [--min-run 100] [--axis FILE [--dir 1]] [--chunk 64] [--threads 16] [--per-wrap] [--cache DIR]\n"); return 2; }
    const double lum = atof(argv[4]), range_um = atof(opt(argc, argv, "--range", "30")), gap_mm = atof(opt(argc, argv, "--gap", "1")), min_um = atof(opt(argc, argv, "--min-run", "100"));
    const int C = atoi(opt(argc, argv, "--chunk", "64")), nt = atoi(opt(argc, argv, "--threads", "16")), per = arg_index(argc, argv, "--per-wrap") > 0;
    if (!(lum > 0) || !(range_um > 0) || !(gap_mm > 0) || C < 1 || nt < 1) { fprintf(stderr, "wrapswitch: the codes key is its voxel size in um (e.g. 4.8); --range, --gap, --chunk > 0\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!z) { fprintf(stderr, "wrapswitch: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(z);
    int nw = 0, capw = 64; ws_wrap *Wr = calloc(capw, sizeof *Wr); double wum = 0;
    DIR *dd = opendir(argv[2]); struct dirent *de; if (!dd) { fprintf(stderr, "wrapswitch: cannot open %s\n", argv[2]); return 1; }
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3; double um = 0;
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        { FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); const char *q = strstr(b, "\"voxel_size_um\":"); if (q) um = atof(q + 16); } }
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = wd_read(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = wd_read(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = wd_read(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(um > 0) || (wum > 0 && fabs(um - wum) > 1e-9)) { free(X); free(Y); free(Z); continue; }
        wum = um;
        if (nw == capw) { capw *= 2; Wr = realloc(Wr, capw * sizeof *Wr); }
        Wr[nw++] = (ws_wrap){w, h, atoi(de->d_name + 1), X, Y, Z, calloc((size_t)w * h, 1), calloc((size_t)w * h, sizeof(float))};
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "wrapswitch: no wraps in %s\n", argv[2]); return 1; }
    const double sc = wum / lum, R = range_um / lum;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
        if (Wr[k].Z[i] < 0) continue;
        const double v[3] = {Wr[k].Z[i] * sc, Wr[k].Y[i] * sc, Wr[k].X[i] * sc};
        for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
    }
    axis ax = {0}; const char *axp = opt(argc, argv, "--axis", nullptr); const int dir = atoi(opt(argc, argv, "--dir", "1"));
    if (axp && axis_load(&ax, axp)) { fprintf(stderr, "wrapswitch: cannot load the axis %s\n", axp); return 1; }
    const double s0 = wum / 2.4;   /* wrap voxels -> level-0 (axis) voxels */
    const int Ri = (int)ceil(R) + 1; int64_t bo[3], bn[3];
    for (int d = 0; d < 3; d++) { bo[d] = (int64_t)floor(lo[d]) - Ri; if (bo[d] < 0) bo[d] = 0; int64_t e = (int64_t)ceil(hi[d]) + Ri + 1; if (e > zm->shape[d]) e = zm->shape[d]; bn[d] = e - bo[d]; }
    long *rs = malloc((nw + 1) * sizeof *rs); rs[0] = 0; for (int k = 0; k < nw; k++) rs[k + 1] = rs[k] + Wr[k].h;
    const size_t plane = (size_t)bn[1] * bn[2]; uint8_t *lab = malloc((size_t)(C + 2 * Ri) * plane);
    if (!lab) { fprintf(stderr, "wrapswitch: no memory for %d rows of %zu voxels (lower --chunk)\n", C + 2 * Ri, plane); return 1; }
    for (int64_t z0 = bo[0]; z0 < bo[0] + bn[0]; z0 += C) {
        const int64_t a = z0 - Ri < bo[0] ? bo[0] : z0 - Ri, b = z0 + C + Ri > bo[0] + bn[0] ? bo[0] + bn[0] : z0 + C + Ri;
        ws_job J = {Wr, nw, rs, rs[nw], sc, R, lab, {a, bo[1], bo[2]}, {b - a, bn[1], bn[2]}, z0, z0 + C, 0, lum, axp ? &ax : nullptr, s0};
        if (z3_read(z, J.o, J.n, lab, nt)) { fprintf(stderr, "wrapswitch: %s\n", z3_error()); return 1; }
        pthread_t th[64]; const int k = nt < 64 ? nt : 64; for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, ws_rows, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    }
    free(lab); z3_close(z);
    /* --axis FILE [--dir 1]: is each coded vertex on its wrap's sheet? A wrap k is U = k + dir theta / 2 pi; a code's q is the published
       sheet's U (+ one constant), so e = q - dir theta / 2 pi - k is the same for every vertex of every wrap traced on the right
       sheet (up to one global gauge), and off by whole turns elsewhere. Gauge: the circular mean of e's fraction, then the most
       common whole turn; right = |e - gauge| < 0.5 turn (mod 14). Over all wraps with one gauge, and per wrap with its own (on its
       majority sheet: coherent though maybe a turn off) */
    double gre = 0, gim = 0; size_t ghist[14] = {0}, nright = 0, nown = 0, ncod = 0;
    float *dall = malloc(sizeof(float) * 1024); size_t nd = 0, capd = 1024; size_t d10 = 0, d20 = 0, sdpos = 0, sdneg = 0; double sdsum = 0;
    if (axp) for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {   /* the global fraction */
        if (!Wr[k].code[i]) continue;
        double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
        const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
        gre += cos(2 * M_PI * e); gim += sin(2 * M_PI * e);
    }
    const double gfr = atan2(gim, gre) / (2 * M_PI);
    if (axp) for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
        if (!Wr[k].code[i]) continue;
        double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
        const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
        ghist[(((int)lround(e - gfr)) % 14 + 14) % 14]++;
    }
    int gm = 0; for (int b = 1; b < 14; b++) if (ghist[b] > ghist[gm]) gm = b;
    ws_stat A[2] = {{0}}; size_t coded = 0, valid = 0;
    if (per) printf("  wrap   coded   theta: sw/cm  run mm   z: sw/cm  run mm%s\n", axp ? "   right  own-sheet" : "");
    for (int k = 0; k < nw; k++) {
        ws_stat s[2] = {{0}}; size_t cd = 0, vl = 0;
        size_t wr = 0, wo = 0, wc = 0, h14[14] = {0};
        for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
            vl += Wr[k].Z[i] >= 0; cd += Wr[k].code[i] > 0; if (!Wr[k].code[i]) continue;
            if (nd == capd) { capd *= 2; dall = realloc(dall, capd * sizeof *dall); }
            const float ad = fabsf(Wr[k].dist[i]); sdsum += Wr[k].dist[i]; sdpos += Wr[k].dist[i] > 0; sdneg += Wr[k].dist[i] < 0;
            dall[nd++] = ad; d10 += ad <= 10; d20 += ad <= 20;
            if (!axp) continue;
            double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
            const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
            const int b = (((int)lround(e - gfr)) % 14 + 14) % 14; h14[b]++; wc++;
            double dg = fmod(e - gfr - gm + 7 + 1400, 14) - 7; wr += fabs(dg) < 0.5;
        }
        if (axp && wc) { int m = 0; for (int b = 1; b < 14; b++) if (h14[b] > h14[m]) m = b; wo = h14[m]; nright += wr; nown += wo; ncod += wc; }
        const int mx = Wr[k].w > Wr[k].h ? Wr[k].w : Wr[k].h; int *sb = malloc((mx + 1) * sizeof *sb); double *lb = malloc((mx + 1) * sizeof *lb);
        for (int d = 0; d < 2; d++) { ws_walk(&Wr[k], d, wum, gap_mm * 1000, min_um, &s[d], sb, lb); A[d].path += s[d].path; A[d].sw += s[d].sw; A[d].runs += s[d].runs; }
        free(sb); free(lb); coded += cd; valid += vl;
        if (per && s[0].path > 0) printf("  w%03d  %5.3f   %10.2f  %6.2f   %8.2f  %6.2f   %5.3f  %5.3f\n", Wr[k].id, vl ? (double)cd / vl : 0, s[0].sw / (s[0].path * 1e-4),
                                          s[0].runs ? s[0].path / s[0].runs * 1e-3 : 0, s[1].path > 0 ? s[1].sw / (s[1].path * 1e-4) : 0, s[1].runs ? s[1].path / s[1].runs * 1e-3 : 0,
                                          wc ? (double)wr / wc : 0, wc ? (double)wo / wc : 0);
        free(Wr[k].dist);
        free(Wr[k].X); free(Wr[k].Y); free(Wr[k].Z); free(Wr[k].code);
    }
    printf("wrapswitch: %d wraps, %zu vertices, %.3f coded (nearest label within %.0f um along the normal); %.1f cm of coded path around "
           "the axis, %.0f switches: %.2f per cm, mean switch-free run %.2f mm; along z %.1f cm, %.0f switches: %.2f per cm, run %.2f mm\n",
           nw, valid, valid ? (double)coded / valid : 0, range_um, A[0].path * 1e-4, A[0].sw, A[0].path > 0 ? A[0].sw / (A[0].path * 1e-4) : 0,
           A[0].runs ? A[0].path / A[0].runs * 1e-3 : 0, A[1].path * 1e-4, A[1].sw, A[1].path > 0 ? A[1].sw / (A[1].path * 1e-4) : 0, A[1].runs ? A[1].path / A[1].runs * 1e-3 : 0);
    if (nd) { qsort(dall, nd, sizeof(float), cmp_fl);
        printf("wrapswitch: distance along the normal to the nearest published sheet (coded vertices): median %.1f um, p90 %.1f; of all vertices <= 10 um %.3f, <= 20 um %.3f\n",
               dall[nd / 2], dall[nd * 9 / 10], valid ? (double)d10 / valid : 0, valid ? (double)d20 / valid : 0);
        if (axp) printf("wrapswitch: signed (+: the published sheet lies outward of the vertex, away from the axis): mean %+.2f um, outward %.3f, inward %.3f, at 0 %.3f\n",
                        sdsum / nd, (double)sdpos / nd, (double)sdneg / nd, (double)(nd - sdpos - sdneg) / nd); }
    if (axp && ncod) printf("wrapswitch: on the right sheet (one global gauge %.3f: q = k + dir theta / 2 pi + gauge, mod 14) %.3f of coded vertices; on their wrap's majority sheet %.3f\n", gfr + gm, (double)nright / ncod, (double)nown / ncod);
    free(dall); free(Wr); free(rs);
    return 0;
}

/* ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U [--range 40] [--min 100] [--median 2] [--measure T] [--threads 16] [--cache DIR]
   Refines tifxyz surfaces (every <wraps-in>/wNNN) onto the papyrus. A fitted winding's level set W = k is smooth at the fit's lattice
   scale (77 um at 19.2 um) and crosses the gaps of wavy sheets: on a 3.9 cm Paris4 slab ~half its 2.4 um render samples were void,
   for a fit to the ground-truth phase too. Each vertex moves along its surface normal (from the grid) to the nearest local maximum
   >= --min of a volume (CT, or a recto probability) within +- --range um, sampled trilinearly every half volume voxel; the
   displacements are median-filtered over (2 median + 1)^2 vertices (a stray peak cannot tear the surface; a vertex without a
   maximum takes its neighbours' median). Vertices are in the wraps' voxel units (meta.json voxel_size_um), the volume at --vol-um.
   Tiles of vertices read only their own box. --measure T: only report the volume at the vertices (mean, share below T; e.g. CT
   < 90: void), writing nothing (<wraps-out> unused). */
typedef struct { const float *X, *Y, *Z, *N; float *disp; int w, r0, r1, c0, c1; const uint8_t *vol; int64_t o[3], n[3]; double s, R, vmin; _Atomic int next; int measure; } ssn_job;
static void *ssn_rows(void *arg) {
    ssn_job *J = arg;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * J->w + c; if (J->Z[i] < 0) continue;
        const double P[3] = {J->Z[i] * J->s - J->o[0], J->Y[i] * J->s - J->o[1], J->X[i] * J->s - J->o[2]};
        if (J->measure) { J->disp[i] = trilin(J->vol, J->n, P[0], P[1], P[2]); continue; }   /* --measure: the value at the vertex */
        if (isnan(J->N[3 * i])) continue;
        const float *N = &J->N[3 * i]; const double st = 0.5;
#define SSV(t_) trilin(J->vol, J->n, P[0] + (t_) * N[0], P[1] + (t_) * N[1], P[2] + (t_) * N[2])
        double bd = 1e300, bt = 0; float vm = SSV(-J->R - st), v0 = SSV(-J->R);
        for (double t = -J->R; t <= J->R; t += st) {
            const float vp = SSV(t + st);
            if (v0 >= vm && v0 > vp && v0 >= J->vmin && fabs(t) < bd) { bd = fabs(t); bt = t; }
            vm = v0; v0 = vp;
        }
#undef SSV
        if (bd < 1e300) J->disp[i] = (float)(bt / J->s);   /* in the wraps' voxels */
    }
    return nullptr;
}
static int cmd_surfsnap(int argc, char **argv) {
    const int vi = arg_index(argc, argv, "--volume");
    if (argc < 4 || vi < 0 || vi + 2 >= argc) { fprintf(stderr, "usage: ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U [--range 40] [--min 100] [--median 2] [--threads 16] [--cache DIR]\n"); return 2; }
    const double vum = atof(opt(argc, argv, "--vol-um", "0")), range_um = atof(opt(argc, argv, "--range", "40")), vmin = atof(opt(argc, argv, "--min", "100"));
    const int nt = atoi(opt(argc, argv, "--threads", "16")), MR = atoi(opt(argc, argv, "--median", "2"));
    if (!(vum > 0) || !(range_um > 0) || MR < 0 || MR > 5 || nt < 1) { fprintf(stderr, "surf-snap: --vol-um U (the volume's voxel um) > 0, --range > 0, --median 0..5\n"); return 2; }
    z3 *vz = z3_open(store_open(argv[vi + 1]), argv[vi + 2], opt(argc, argv, "--cache", nullptr)); if (!vz) { fprintf(stderr, "surf-snap: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(vz);
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (atof(opt(argc, argv, "--measure", "0")) <= 0 && system(cmd)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "surf-snap: cannot open %s\n", argv[2]); return 1; }
    struct dirent *de; int nw = 0; size_t nmoved = 0, nvert = 0, mlow = 0, mn = 0; double sabs = 0, msum = 0;
    const double measure = atof(opt(argc, argv, "--measure", "0"));   /* > 0: report the share of vertices where the volume is below it */
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3; double wum = 0, scale = 0; int winding = 0;
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        { FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); }
          const char *q = strstr(b, "\"voxel_size_um\":"); if (q) wum = atof(q + 16);
          q = strstr(b, "\"scale\":"); if (q) { q = strchr(q, '['); if (q) scale = atof(q + 1); }
          q = strstr(b, "\"winding\":"); if (q) winding = atoi(q + 10); }
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = read_tif_f32(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = read_tif_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = read_tif_f32(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(wum > 0) || !(scale > 0) || w < 3 || h < 3) { free(X); free(Y); free(Z); continue; }
        const size_t nv = (size_t)w * h; const double s = wum / vum, R = range_um / vum;
        float *N = malloc(nv * 3 * sizeof(float)), *disp = malloc(nv * 4), *med = malloc(nv * 4);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {   /* unit normals (central differences where the neighbours exist) */
            const size_t i = (size_t)r * w + c; float *n3 = &N[3 * i]; n3[0] = n3[1] = n3[2] = NAN; disp[i] = NAN; if (Z[i] < 0) continue;
            const int cl = c > 0 && Z[i - 1] >= 0 ? c - 1 : c, cr = c + 1 < w && Z[i + 1] >= 0 ? c + 1 : c;
            const int ru = r > 0 && Z[i - w] >= 0 ? r - 1 : r, rd = r + 1 < h && Z[i + w] >= 0 ? r + 1 : r;
            if (cl == cr || ru == rd) continue;
            const size_t a = (size_t)r * w + cl, b = (size_t)r * w + cr, u = (size_t)ru * w + c, d = (size_t)rd * w + c;
            const double du[3] = {Z[b] - Z[a], Y[b] - Y[a], X[b] - X[a]}, dv[3] = {Z[d] - Z[u], Y[d] - Y[u], X[d] - X[u]};
            const double q[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]}, ql = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            if (ql > 0) for (int k = 0; k < 3; k++) n3[k] = (float)(q[k] / ql);
        }
        const int TR = 128, TC = 512;
        for (int r0 = 0; r0 < h; r0 += TR) for (int c0 = 0; c0 < w; c0 += TC) {
            const int r1 = r0 + TR < h ? r0 + TR : h, c1 = c0 + TC < w ? c0 + TC : w;
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
                const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
                const double v[3] = {Z[i] * s, Y[i] * s, X[i] * s}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
            }
            if (lo[0] > hi[0]) continue;
            ssn_job J = {X, Y, Z, N, disp, w, r0, r1, c0, c1, nullptr, {0, 0, 0}, {0, 0, 0}, s, R, vmin, 0, measure > 0}; int ok = 1;
            for (int d = 0; d < 3; d++) {
                int64_t a = (int64_t)floor(lo[d] - R - 2), b = (int64_t)ceil(hi[d] + R + 2) + 1;
                if (a < 0) a = 0;
                if (b > zm->shape[d]) b = zm->shape[d];
                if (b <= a) ok = 0;
                J.o[d] = a; J.n[d] = b - a;
            }
            const size_t bytes = (size_t)J.n[0] * J.n[1] * J.n[2];
            if (!ok || bytes > ((size_t)4 << 30)) { if (ok) fprintf(stderr, "surf-snap: %s tile %d,%d spans %lldx%lldx%lld voxels (a fold?): skipped\n", de->d_name, r0, c0, (long long)J.n[0], (long long)J.n[1], (long long)J.n[2]); continue; }
            uint8_t *vol = malloc(bytes);
            if (!vol || z3_read(vz, J.o, J.n, vol, nt)) { free(vol); fprintf(stderr, "surf-snap: %s\n", z3_error()); continue; }
            J.vol = vol;
            pthread_t th[64]; const int k = nt < 64 ? nt : 64;
            for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, ssn_rows, &J);
            for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
            free(vol);
        }
        if (measure > 0) {   /* --measure T: the volume at the vertices, no output */
            for (size_t i = 0; i < nv; i++) if (Z[i] >= 0 && !isnan(disp[i])) { msum += disp[i]; mlow += disp[i] < measure; mn++; }
            nw++; free(X); free(Y); free(Z); free(N); free(disp); free(med); continue;
        }
        float win[121]; size_t moved = 0, valid = 0;
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; med[i] = NAN; if (Z[i] < 0) continue;
            valid++; int k = 0;
            for (int a = r - MR; a <= r + MR; a++) for (int b = c - MR; b <= c + MR; b++)
                if (a >= 0 && a < h && b >= 0 && b < w && !isnan(disp[(size_t)a * w + b])) win[k++] = disp[(size_t)a * w + b];
            if (k >= (2 * MR + 1) * (2 * MR + 1) / 5 + 1) { qsort(win, k, sizeof(float), cmp_fl); med[i] = win[k / 2]; }
        }
        for (size_t i = 0; i < nv; i++) {
            if (Z[i] < 0 || isnan(med[i]) || isnan(N[3 * i])) continue;
            Z[i] += med[i] * N[3 * i]; Y[i] += med[i] * N[3 * i + 1]; X[i] += med[i] * N[3 * i + 2]; moved++; sabs += fabs(med[i]) * wum;
        }
        snprintf(p, sizeof p, "%s/%s", argv[3], de->d_name);
        if (tifxyz_write(p, w, h, X, Y, Z, 1 / scale, wum, winding, "ufsm surf-snap (wraps refined onto the papyrus)")) { fprintf(stderr, "surf-snap: cannot write %s\n", p); return 1; }
        nw++; nmoved += moved; nvert += valid;
        free(X); free(Y); free(Z); free(N); free(disp); free(med);
    }
    closedir(dd); z3_close(vz);
    if (measure > 0) { printf("surf-snap --measure: %d wraps, %zu vertices: mean %.1f, below %g %.3f\n", nw, mn, mn ? msum / mn : 0, measure, mn ? (double)mlow / mn : 0); return nw ? 0 : 1; }
    fprintf(stderr, "surf-snap: %d wraps, %zu of %zu vertices moved (mean |shift| %.1f um) -> %s\n", nw, nmoved, nvert, nmoved ? sabs / nmoved : 0, argv[3]);
    return nw ? 0 : 1;
}

/* ufsm field-blend OUT IN1 IN2 ... [--weights a,b,...] [--threads N]: fit-field sets of several models on one grid blended into
   OUT (src/fblend.h: the phase as the confidence-weighted complex mean, its length the new confidence) */
static int cmd_field_blend(int argc, char **argv) {
    const char *in[32]; float w[32]; int n = 0; const char *ws = nullptr, *out = nullptr; int nt = 8;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) { ws = argv[++i]; continue; }
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) { nt = atoi(argv[++i]); continue; }
        if (!strncmp(argv[i], "--", 2)) { fprintf(stderr, "field-blend: unknown option %s\n", argv[i]); return 2; }
        if (!out) out = argv[i]; else if (n < 32) in[n++] = argv[i]; else { fprintf(stderr, "field-blend: at most 32 inputs\n"); return 2; }
    }
    if (!out || n < 1) { fprintf(stderr, "usage: ufsm field-blend <out> <fields1> [<fields2> ...] [--weights a,b,...] [--threads N]\n"); return 2; }
    if (ws) {
        int k = 0; char *t = strdup(ws);
        for (char *q = strtok(t, ","); q; q = strtok(nullptr, ",")) { if (k < 32) w[k] = (float)atof(q); k++; }
        free(t);
        if (k != n) { fprintf(stderr, "field-blend: %d weights for %d inputs\n", k, n); return 2; }
    }
    return fblend_run(out, n, in, ws ? w : nullptr, nt < 1 ? 1 : nt);
}
/* ufsm fit-compare <reference-fields> <fields>: how a field set (e.g. a network's, predict --fit-zarr) agrees with a reference
   (e.g. the oracle, band --fit-fields) cell by cell; both must cover the same physical cells (origin x um, h x um equal), at any
   native voxel sizes. Over cells where the reference phase confidence is > 0.3 and both have material: phase |error| (turns),
   phase confidence, sheet spacing ratio (um / um), normal angle (principal eigenvectors of n n^T), and the share of gross
   phase errors (> 1/4 turn) by reference spacing and by the fields' confidence. Histograms, no per-cell arrays. */
static void fc_axis(const fit_fields *f, size_t i, double v[3]) {   /* principal eigenvector of the cell's n n^T (power iteration) */
    double T[3][3]; const double t[6] = {f->t[0][i] / 255.0, f->t[1][i] / 255.0, f->t[2][i] / 255.0, f->t[3][i] / 255.0 - 0.5, f->t[4][i] / 255.0 - 0.5, f->t[5][i] / 255.0 - 0.5};
    T[0][0] = t[0]; T[1][1] = t[1]; T[2][2] = t[2]; T[0][1] = T[1][0] = t[3]; T[0][2] = T[2][0] = t[4]; T[1][2] = T[2][1] = t[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;   /* start: the largest diagonal axis */
    for (int it = 0; it < 24; it++) {
        double w[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) w[a] += T[a][b] * v[b];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-12) return; for (int a = 0; a < 3; a++) v[a] = w[a] / l;
    }
}
static double hist_q(const size_t *h, int nb, double lo, double w, double q) {   /* quantile of a histogram (bin centres) */
    size_t n = 0; for (int k = 0; k < nb; k++) n += h[k];
    size_t c = 0; for (int k = 0; k < nb; k++) { c += h[k]; if (n && (double)c >= q * n) return lo + (k + 0.5) * w; }
    return lo + nb * w;
}
static int cmd_fit_compare(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fit-compare <reference-fields> <fields>\n"); return 2; }
    fit_fields_buf A, B;
    if (fit_fields_load(argv[2], 0, 0, 1, &A) || fit_fields_load(argv[3], 0, 0, 1, &B)) { fprintf(stderr, "fit-compare: cannot load the fields\n"); return 1; }
    /* the candidate's cells may be k times the reference's (a coarser net, e.g. 38.4 um cells against the 19.2 um oracle): each
       reference cell reads the candidate cell that contains it */
    const int k = (int)lround(B.h * B.um / (A.h * A.um));
    for (int d = 0; d < 3; d++) {
        if (k < 1 || fabs(k * A.h * A.um - B.h * B.um) > 1e-6 * B.h * B.um || (A.shape[d] + k - 1) / k != B.shape[d] ||
            fabs(A.origin[d] * A.um - B.origin[d] * B.um) > 0.5 * A.h * A.um) {
            fprintf(stderr, "fit-compare: the field sets cover different cells (shape %d/%d, origin %.1f/%.1f um, cell %.2f/%.2f um on axis %d)\n", A.shape[d], B.shape[d], A.origin[d] * A.um, B.origin[d] * B.um, A.h * A.um, B.h * B.um, d); return 2;
        }
    }
    const size_t n = (size_t)A.shape[0] * A.shape[1] * A.shape[2];
    enum { NP = 500, NC = 100, NR = 200, NA = 900, NS = 5, NK = 4 };
    size_t hp[NP] = {0}, hca[NC] = {0}, hcb[NC] = {0}, hr[NR] = {0}, ha[NA] = {0}, sn[NS] = {0}, sg[NS] = {0}, kn[NK] = {0}, kg[NK] = {0}, m = 0, lt01 = 0, gt25 = 0;
    const double sb[NS + 1] = {0, 100, 140, 180, 240, 1e30}, cb[NK + 1] = {0, 0.2, 0.4, 0.6, 1.01};
    if (k > 1) printf("fit-compare: candidate cells %dx the reference's\n", k);
    for (size_t i = 0; i < n; i++) {
        const size_t az = i / ((size_t)A.shape[1] * A.shape[2]), ay = i / A.shape[2] % A.shape[1], ax = i % A.shape[2];
        const size_t j = ((az / k) * B.shape[1] + ay / k) * B.shape[2] + ax / k;   /* the candidate cell */
        if (!A.f.w[i] || !B.f.w[j]) continue;
        const double ac = (A.f.pc[i] - 128) / 127.0, as = (A.f.ps[i] - 128) / 127.0, bc = (B.f.pc[j] - 128) / 127.0, bs = (B.f.ps[j] - 128) / 127.0;
        const double ca = hypot(ac, as), cbv = hypot(bc, bs); if (ca <= 0.3) continue;
        m++;
        double e = (atan2(bs, bc) - atan2(as, ac)) / (2 * M_PI); e -= round(e); e = fabs(e);
        hp[(int)fmin(NP - 1, e / 0.5 * NP)]++; lt01 += e < 0.1; gt25 += e > 0.25;
        hca[(int)fmin(NC - 1, ca * NC)]++; hcb[(int)fmin(NC - 1, cbv * NC)]++;
        const double spa = A.f.sp[i] * A.h / 8.0 * A.um, spb = B.f.sp[j] * B.h / 8.0 * B.um;   /* spacing in um (sp = 8 / (h rho)) */
        if (spa > 0 && spb > 0) hr[(int)fmin(NR - 1, fmax(0, (log2(spb / spa) + 2) / 4 * NR))]++;   /* log2 ratio in [-2, 2) */
        for (int k = 0; k < NS; k++) if (spa >= sb[k] && spa < sb[k + 1]) { sn[k]++; sg[k] += e > 0.25; }
        for (int k = 0; k < NK; k++) if (cbv >= cb[k] && cbv < cb[k + 1]) { kn[k]++; kg[k] += e > 0.25; }
        double va[3], vb[3]; fc_axis(&A.f, i, va); fc_axis(&B.f, j, vb);
        const double ang = acos(fmin(1, fabs(va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]))) * 180 / M_PI;
        ha[(int)fmin(NA - 1, ang / 90 * NA)]++;
    }
    if (!m) { fprintf(stderr, "fit-compare: no cells with reference phase\n"); return 1; }
    printf("fit-compare: %zu cells with reference phase (confidence > 0.3) and material in both\n", m);
    printf("  phase |error| (turns): median %.3f p75 %.3f p90 %.3f; < 0.1 %.3f, > 0.25 %.3f\n", hist_q(hp, NP, 0, 0.5 / NP, 0.5), hist_q(hp, NP, 0, 0.5 / NP, 0.75), hist_q(hp, NP, 0, 0.5 / NP, 0.9), (double)lt01 / m, (double)gt25 / m);
    printf("  phase confidence: median %.2f (reference %.2f)\n", hist_q(hcb, NC, 0, 1.0 / NC, 0.5), hist_q(hca, NC, 0, 1.0 / NC, 0.5));
    printf("  spacing ratio: median %.3f p10 %.3f p90 %.3f\n", exp2(hist_q(hr, NR, -2, 4.0 / NR, 0.5)), exp2(hist_q(hr, NR, -2, 4.0 / NR, 0.1)), exp2(hist_q(hr, NR, -2, 4.0 / NR, 0.9)));
    printf("  normal angle (degrees): median %.1f p90 %.1f\n", hist_q(ha, NA, 0, 90.0 / NA, 0.5), hist_q(ha, NA, 0, 90.0 / NA, 0.9));
    printf("  gross phase errors by reference spacing:");
    for (int k = 0; k < NS; k++) printf("  %s%.0f um %.3f (%zu)", k == NS - 1 ? ">= " : "< ", k == NS - 1 ? sb[k] : sb[k + 1], sn[k] ? (double)sg[k] / sn[k] : 0, sn[k]);
    printf("\n  gross phase errors by confidence:");
    for (int k = 0; k < NK; k++) printf("  %.1f-%.1f %.3f (%zu)", cb[k], fmin(1, cb[k + 1]), kn[k] ? (double)kg[k] / kn[k] : 0, kn[k]);
    printf("\n");
    {   /* the turn count without the fitter: 0.5 mm segments along the radial direction (away from the axis, in z planes) from
           every 4th reference cell; the reference crosses n turns (its phase unwrapped along the segment, every step known with
           confidence > 0.3); the candidate's count from its phase unwrapped the same way, and from its density integrated along
           the segment (|n . r| / spacing, n the principal axis of its n n^T): a field too coarse for its phase to resolve the
           sheets can still count them by density. Error = candidate - reference turns (negative: sheets missed) */
        const double L = 500.0, ds = 0.5 * A.h * A.um;   /* segment length and step (um) */
        const int ns = (int)(L / ds);
        enum { NE = 400 }; size_t hep[NE] = {0}, hed[NE] = {0}, nseg = 0, slp = 0, sld = 0; double bp = 0, bd = 0, rt = 0;
        size_t pk_n = 0, pk_sp = 0, pk_sd = 0;   /* segments whose reference spacing is under 100 um: slips */
        const double cu = A.h * A.um, ccu = B.h * B.um;   /* reference / candidate cell edge (um) */
        for (int z = 0; z < A.shape[0]; z += 4) for (int y = 0; y < A.shape[1]; y += 4) for (int x = 0; x < A.shape[2]; x += 4) {
            const double nz = A.h * (z + 0.5);   /* native z of the cell centre (reference's native voxels, box-local) */
            const int az = (int)fmin(fmax(nz, 0), A.shape[0] * A.h); const double cyv = A.f.cy[az], cxv = A.f.cx[az];
            double py = A.h * (y + 0.5), px = A.h * (x + 0.5);
            double ry = py - cyv, rx = px - cxv; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            double pa = 0, pb = 0, ta = 0, tb = 0, td = 0, la = 0, lb = 0, spsum = 0; int ok = 1, hb = 0;
            for (int k = 0; k <= ns && ok; k++) {
                const double qy = py + ry * k * ds / A.um, qx = px + rx * k * ds / A.um;   /* native voxels */
                const int iy = (int)(qy / A.h), ix = (int)(qx / A.h);
                if (iy < 0 || ix < 0 || iy >= A.shape[1] || ix >= A.shape[2]) { ok = 0; break; }
                const size_t i = ((size_t)z * A.shape[1] + iy) * A.shape[2] + ix;
                const double ac = (A.f.pc[i] - 128) / 127.0, as = (A.f.ps[i] - 128) / 127.0;
                if (!A.f.w[i] || hypot(ac, as) <= 0.3) { ok = 0; break; }
                const size_t j = (((size_t)(nz * A.um / ccu)) * B.shape[1] + (size_t)(qy * A.um / ccu)) * B.shape[2] + (size_t)(qx * A.um / ccu);
                const double bc = (B.f.pc[j] - 128) / 127.0, bs = (B.f.ps[j] - 128) / 127.0;
                pa = atan2(as, ac) / (2 * M_PI); spsum += A.f.sp[i] * cu / 8.0;
                if (k) { double d = pa - la; d -= round(d); ta += d; }
                la = pa;
                if (hypot(bc, bs) > 1e-3) { pb = atan2(bs, bc) / (2 * M_PI); if (hb) { double d = pb - lb; d -= round(d); tb += d; } lb = pb; hb = 1; }
                if (k < ns && B.f.sp[j] && B.f.w[j]) {   /* density: |n . r| / spacing over the step */
                    double v[3]; fc_axis(&B.f, j, v);
                    td += fabs(v[1] * ry + v[2] * rx) * ds / (B.f.sp[j] * ccu / 8.0);
                }
            }
            if (!ok) continue;
            const double ref = fabs(ta); if (ref < 0.5) continue;   /* a segment with under half a turn says nothing about counting */
            const double ep = fabs(tb) - ref, ed = td - ref;
            hep[(int)fmin(NE - 1, fmax(0, (ep + 4) / 8 * NE))]++; hed[(int)fmin(NE - 1, fmax(0, (ed + 4) / 8 * NE))]++;
            nseg++; slp += fabs(ep) >= 0.5; sld += fabs(ed) >= 0.5; bp += ep; bd += ed; rt += ref;
            if (spsum / (ns + 1) < 100) { pk_n++; pk_sp += fabs(ep) >= 0.5; pk_sd += fabs(ed) >= 0.5; }
        }
        if (nseg) {
            printf("  turn counts on %zu radial 0.5 mm segments (reference %.2f turns each on average):\n", nseg, rt / nseg);
            printf("    by phase:   error median %+.2f, mean %+.2f turns, off by >= 0.5 turn %.3f (packed < 100 um: %.3f of %zu)\n",
                   hist_q(hep, NE, -4, 8.0 / NE, 0.5), bp / nseg, (double)slp / nseg, pk_n ? (double)pk_sp / pk_n : 0, pk_n);
            printf("    by density: error median %+.2f, mean %+.2f turns, off by >= 0.5 turn %.3f (packed < 100 um: %.3f)\n",
                   hist_q(hed, NE, -4, 8.0 / NE, 0.5), bd / nseg, (double)sld / nseg, pk_n ? (double)pk_sd / pk_n : 0);
        }
    }
    fit_fields_free(&A); fit_fields_free(&B);
    return 0;
}

/* ufsm gap-compare <array-root> <array-key> <reference-fields> [--net FIELDS] [--R-um 115] [--thr 128] [--min-run 2] [--origin z,y,x]
   Sheet density from gaps in a uint8 array (a papyrus / void probability from predict, or a raw CT level; the key is its voxel
   size in um, positions mapped physically; origin from --origin, ufsm.origin_zyx, else 0): per reference field cell with phase
   (confidence > 0.3) the radial ray through the cell centre at its z, +-R um, binarised at --thr, void -> papyrus entries counted (runs shorter than --min-run ignored) per
   2R voxels; against the reference's sheet density (1 / spacing), by reference spacing; --net: the same for a field set's
   density (e.g. a winding network's, any native voxel size covering the same cells). Medians / quantiles of the ratios. */
static double axis_lerp_tab(const float *v, int n, double z) {   /* a per-native-z axis table at fractional z (clamped) */
    if (z <= 0) return v[0];
    if (z >= n - 1) return v[n - 1];
    const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]);
}
typedef struct { const uint8_t *pl; int64_t H, W; const fit_fields_buf *A, *Bn; int fz, R, thr, minrun; double s; long long po[3]; size_t (*hg)[5][200], (*hn)[5][200]; pthread_mutex_t *mu; _Atomic int next; } gap_job;
static void *gap_rows(void *arg) {
    gap_job *J = arg; const fit_fields *f = &J->A->f; const int na = (int)(f->D * f->h) + 1;
    size_t lg[5][200] = {{0}}, ln[5][200] = {{0}};
    for (int fy; (fy = atomic_fetch_add(&J->next, 1)) < f->H;) for (int fx = 0; fx < f->W; fx++) {
        const size_t i = ((size_t)J->fz * f->H + fy) * f->W + fx;
        if (!f->w[i] || !f->sp[i]) continue;
        const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0; if (hypot(mc, ms) <= 0.3) continue;
        const double pz = (J->fz + 0.5) * f->h, py = (fy + 0.5) * f->h, px = (fx + 0.5) * f->h;
        const double cy = axis_lerp_tab(f->cy, na, pz), cx = axis_lerp_tab(f->cx, na, pz);
        double uy = py - cy, ux = px - cx; const double ul = hypot(uy, ux); if (ul < 1) continue; uy /= ul; ux /= ul;
        const double yp = (py + J->A->origin[1]) * J->s - J->po[1], xp = (px + J->A->origin[2]) * J->s - J->po[2];   /* on the array's grid */
        int prev = -1, run = 0, entries = 0, lastv = -1, len = 0;
        for (int t = -J->R; t <= J->R; t++) {
            const int64_t yy = (int64_t)lround(yp + t * uy), xx = (int64_t)lround(xp + t * ux);
            if (yy < 0 || xx < 0 || yy >= J->H || xx >= J->W) { prev = -1; run = 0; continue; }
            len++;
            const int v = J->pl[(size_t)yy * J->W + xx] >= J->thr;
            if (v == prev) { run++; if (run == J->minrun && v == 1 && lastv == 0) entries++; if (run >= J->minrun) lastv = v; }
            else { prev = v; run = 1; if (J->minrun <= 1) { if (v == 1 && lastv == 0) entries++; lastv = v; } }
        }
        if (len < J->R) continue;
        const double spa = f->sp[i] * f->h / 8.0 * J->A->um, rho_ref = 1.0 / spa;   /* sheets per um */
        const double rho_gap = entries / (len * J->A->um / J->s);
        int sbin = spa < 100 ? 0 : spa < 140 ? 1 : spa < 180 ? 2 : spa < 240 ? 3 : 4;
        if (rho_gap > 0) lg[sbin][(int)fmin(199, fmax(0, (log2(rho_gap / rho_ref) + 2) / 4 * 200))]++; else lg[sbin][0]++;
        if (J->Bn && J->Bn->f.sp[i]) { const double spb = J->Bn->f.sp[i] * J->Bn->h / 8.0 * J->Bn->um; ln[sbin][(int)fmin(199, fmax(0, (log2(spa / spb) + 2) / 4 * 200))]++; }
    }
    pthread_mutex_lock(J->mu);
    for (int a = 0; a < 5; a++) for (int b = 0; b < 200; b++) { (*J->hg)[a][b] += lg[a][b]; (*J->hn)[a][b] += ln[a][b]; }
    pthread_mutex_unlock(J->mu);
    return nullptr;
}
static int cmd_gap_compare(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm gap-compare <array-root> <array-key> <reference-fields> [--net FIELDS] [--R-um 115] [--thr 128] [--min-run 2] [--origin z,y,x] [--um U] [--cache DIR] [--threads 16]\n"); return 2; }
    const double aum = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : atof(argv[3]), Rum = atof(opt(argc, argv, "--R-um", "115"));   /* --um: when the key is not the voxel size */
    const int thr = atoi(opt(argc, argv, "--thr", "128")), minrun = atoi(opt(argc, argv, "--min-run", "2")), nt = atoi(opt(argc, argv, "--threads", "16"));
    if (!(aum > 0)) { fprintf(stderr, "gap-compare: the array key must be its voxel size in um (e.g. 2.4)\n"); return 2; }
    fit_fields_buf A, Bn; const char *net = opt(argc, argv, "--net", nullptr);
    if (fit_fields_load(argv[4], 0, 0, 1, &A) || (net && fit_fields_load(net, 0, 0, 1, &Bn))) { fprintf(stderr, "gap-compare: cannot load the fields\n"); return 1; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)); if (!pz) { fprintf(stderr, "gap-compare: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};   /* the array's origin on its own grid: --origin, else ufsm.origin_zyx (predict outputs), else 0 (whole volumes) */
    if (*opt(argc, argv, "--origin", "")) sscanf(opt(argc, argv, "--origin", ""), "%lld,%lld,%lld", po, po + 1, po + 2);
    else { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
           if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
           const char *q = strstr(jb, "\"origin_zyx\":"); if (q) sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2); }
    const z3_meta *zm = z3_meta_of(pz); const double s = A.um / aum; const int R = (int)lround(Rum / aum);
    /* the planes of the array covering the fields' y / x extent */
    long long lo[2], hi[2];
    for (int d = 1; d < 3; d++) {
        lo[d - 1] = (long long)floor(A.origin[d] * s) - po[d] - R - 2; hi[d - 1] = (long long)ceil((A.origin[d] + A.shape[d] * A.h) * s) - po[d] + R + 2;
        if (lo[d - 1] < 0) lo[d - 1] = 0;
        if (hi[d - 1] > zm->shape[d]) hi[d - 1] = zm->shape[d];
    }
    const int64_t H = hi[0] - lo[0], W = hi[1] - lo[1];
    uint8_t *pl = malloc((size_t)H * W); size_t hg[5][200] = {{0}}, hn[5][200] = {{0}}; pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    const long long pol[3] = {po[0], po[1] + lo[0], po[2] + lo[1]};
    for (int fz = 0; fz < A.shape[0]; fz++) {
        const int64_t z = (int64_t)lround(((fz + 0.5) * A.h + A.origin[0]) * s) - po[0]; if (z < 0 || z >= zm->shape[0]) continue;
        const int64_t o[3] = {z, lo[0], lo[1]}, n[3] = {1, H, W};
        if (z3_read(pz, o, n, pl, nt)) { fprintf(stderr, "gap-compare: %s\n", z3_error()); return 1; }
        gap_job J = {pl, H, W, &A, net ? &Bn : nullptr, fz, R, thr, minrun, s, {pol[0], pol[1], pol[2]}, &hg, &hn, &mu, 0};
        pthread_t th[64]; const int k = nt < 1 ? 1 : nt > 64 ? 64 : nt;
        for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, gap_rows, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    }
    static const char *bn[5] = {"< 100 um", "100-140", "140-180", "180-240", ">= 240"};
    printf("gap-compare: density ratio to the reference (median, p10, p90) by reference spacing; R %.0f um, threshold %d, min run %d\n", Rum, thr, minrun);
    for (int a = 0; a < 5; a++) {
        size_t t = 0; for (int b = 0; b < 200; b++) t += hg[a][b]; if (!t) continue;
        printf("  %-9s %8zu cells: gaps %.2f (%.2f-%.2f)", bn[a], t, exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.5)), exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.1)), exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.9)));
        size_t u = 0; for (int b = 0; b < 200; b++) u += hn[a][b];
        if (u) printf("   net density %.2f (%.2f-%.2f)", exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.5)), exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.1)), exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.9)));
        printf("\n");
    }
    free(pl); z3_close(pz); fit_fields_free(&A); if (net) fit_fields_free(&Bn);
    return 0;
}

/* ufsm gap-density <papyrus-root> <papyrus-key> <fields-in> <fields-out> [--fuse-um 150] [--R-um 115] [--thr 128] [--min-run 2]
   A field set (any native voxel size) with its sheet spacing fused with the papyrus-gap spacing: per cell the radial ray through
   its centre at its z, +-R um, over the papyrus / void probability (task papyrus predict output; its key is its voxel size in um,
   origin_zyx from its attributes), void -> papyrus entries counted (runs shorter than --min-run voxels ignored). Where the
   fields' spacing is under --fuse-um and the gaps' is smaller, the gaps' replaces it: networks under-count packed sheets (block B:
   0.72 of the true density under 100 um) while papyrus gaps count them (1.04) but over-count loose sheets (each sheet's own
   layers and cracks: 1.4-2.2). Writes fields-out as raw fit fields. */
static int cmd_gap_density(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm gap-density <papyrus-root> <papyrus-key> <fields-in> <fields-out> [--fuse-um 150] [--R-um 115] [--thr 128] [--min-run 2]\n"); return 2; }
    const double fuse = atof(opt(argc, argv, "--fuse-um", "150")), Rum = atof(opt(argc, argv, "--R-um", "115")), ump = atof(argv[3]);
    const int thr = atoi(opt(argc, argv, "--thr", "128")), minrun = atoi(opt(argc, argv, "--min-run", "2")), nt = atoi(opt(argc, argv, "--threads", "16"));
    if (!(ump > 0)) { fprintf(stderr, "gap-density: the papyrus key must be its voxel size in um\n"); return 2; }
    fit_fields_buf A; if (fit_fields_load(argv[4], 0, 0, 1, &A)) { fprintf(stderr, "gap-density: cannot load %s\n", argv[4]); return 1; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], nullptr); if (!pz) { fprintf(stderr, "gap-density: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};
    { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
      if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
      const char *q = strstr(jb, "\"origin_zyx\":"); if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2) != 3) { fprintf(stderr, "gap-density: no origin_zyx in %s\n", jp); return 2; } }
    const z3_meta *zm = z3_meta_of(pz); const int64_t PH = zm->shape[1], PW = zm->shape[2];
    const double s = A.um / ump;   /* fields' native voxel -> papyrus voxels */
    const int R = (int)lround(Rum / ump); const int na = (int)(A.f.D * A.f.h) + 1;
    uint8_t *pl = malloc((size_t)PH * PW), *sp = malloc((size_t)A.shape[0] * A.shape[1] * A.shape[2]);
    size_t fused = 0, cells = 0;
    for (int fz = 0; fz < A.shape[0]; fz++) {
        const double zf = (fz + 0.5) * A.h; const int64_t zp = (int64_t)lround((zf + A.origin[0]) * s) - po[0];
        const int have = zp >= 0 && zp < zm->shape[0];
        if (have) { const int64_t o[3] = {zp, 0, 0}, n[3] = {1, PH, PW}; if (z3_read(pz, o, n, pl, nt)) { fprintf(stderr, "gap-density: %s\n", z3_error()); return 1; } }
        const double cy = axis_lerp_tab(A.f.cy, na, zf), cx = axis_lerp_tab(A.f.cx, na, zf);
        for (int fy = 0; fy < A.shape[1]; fy++) for (int fx = 0; fx < A.shape[2]; fx++) {
            const size_t i = ((size_t)fz * A.shape[1] + fy) * A.shape[2] + fx; sp[i] = A.f.sp[i];
            if (!have || !A.f.w[i] || !A.f.sp[i]) continue;
            const double spn = A.f.sp[i] * A.h / 8.0 * A.um; if (spn >= fuse) continue;
            const double yf = (fy + 0.5) * A.h, xf = (fx + 0.5) * A.h; double uy = yf - cy, ux = xf - cx; const double ul = hypot(uy, ux); if (ul < 1) continue; uy /= ul; ux /= ul;
            const double yp = (yf + A.origin[1]) * s - po[1], xp = (xf + A.origin[2]) * s - po[2];
            int prev = -1, run = 0, entries = 0, lastv = -1, len = 0;
            for (int t = -R; t <= R; t++) {
                const int64_t yy = (int64_t)lround(yp + t * uy), xx = (int64_t)lround(xp + t * ux);
                if (yy < 0 || xx < 0 || yy >= PH || xx >= PW) { prev = -1; run = 0; continue; }
                len++;
                const int v = pl[(size_t)yy * PW + xx] >= thr;
                if (v == prev) { run++; if (run == minrun && v == 1 && lastv == 0) entries++; if (run >= minrun) lastv = v; }
                else { prev = v; run = 1; if (minrun <= 1) { if (v == 1 && lastv == 0) entries++; lastv = v; } }
            }
            cells++;
            if (!entries || len < R) continue;
            const double spg = len * ump / entries;   /* um per sheet */
            if (spg < spn) { const double v = spg / A.um / A.h * 8.0; sp[i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); fused++; }
        }
    }
    uint8_t *a[10] = {A.a[0], A.a[1], A.a[2], A.a[3], A.a[4], A.a[5], A.a[6], A.a[7], sp, A.a[9]};
    const long long org[3] = {A.origin[0], A.origin[1], A.origin[2]};
    if (fit_fields_save(argv[5], org, A.shape, A.h, A.um, a, A.cy, A.cx, na)) { fprintf(stderr, "gap-density: cannot write %s\n", argv[5]); return 1; }
    fprintf(stderr, "gap-density: %zu of %zu packed cells took the papyrus-gap spacing -> %s\n", fused, cells, argv[5]);
    free(pl); free(sp); z3_close(pz); fit_fields_free(&A);
    return 0;
}

/* ufsm void-pool <pred-root> <pred-key> <out-group> <out-um> --vol-shape z,y,x [--factor 4] [--thr 128] [--shard 512]
   The aux input for a coarse winding net (train --aux-input): per output voxel the fraction of its factor^3 source voxels that are
   void (papyrus probability < --thr) x 255, from a task papyrus predict output (box-local, ufsm.origin_zyx; on the level-0 grid
   of --vol-shape) into <out-group>/<out-um>, a whole-volume array (shape ceil(vol-shape / factor)) shared by many boxes: each
   output shard the box meets is read, the box's part replaced, and written back. Thin gaps between packed sheets survive as
   void fractions where a 9.6 um CT averages them away. */
static int cmd_void_pool(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm void-pool <pred-root> <pred-key> <out-group> <out-um> --vol-shape z,y,x [--factor 4] [--thr 128] [--shard 512]\n"); return 2; }
    const int F = atoi(opt(argc, argv, "--factor", "4")), thr = atoi(opt(argc, argv, "--thr", "128")), S = atoi(opt(argc, argv, "--shard", "512")), nt = 16;
    long long vs[3]; if (sscanf(opt(argc, argv, "--vol-shape", ""), "%lld,%lld,%lld", vs, vs + 1, vs + 2) != 3 || F < 1 || S < 1) { fprintf(stderr, "void-pool: --vol-shape z,y,x (level 0)\n"); return 2; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], nullptr); if (!pz) { fprintf(stderr, "void-pool: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};
    { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
      if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
      const char *q = strstr(jb, "\"origin_zyx\":"); if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2) != 3) { fprintf(stderr, "void-pool: no origin_zyx in %s\n", jp); return 2; } }
    const z3_meta *pm = z3_meta_of(pz); const double um = atof(argv[5]);
    const int64_t osh[3] = {(vs[0] + F - 1) / F, (vs[1] + F - 1) / F, (vs[2] + F - 1) / F};
    char lv[32], od[1400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    char at[300]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus void fraction x 255 over %d^3 voxels (probability < %d)\",\"factor\":%d}}", F, thr, F);
    z3w *w = z3w_create(od, osh, S, 0, 0, at); if (!w || z3w_write_group(argv[4], &um, 1, "ufsm-void-fraction", nullptr)) { fprintf(stderr, "void-pool: %s\n", z3w_error()); return 1; }
    z3 *oz = z3_open(store_open(argv[4]), lv, nullptr);   /* existing shard contents (other boxes) */
    int64_t lo[3], hi[3]; for (int d = 0; d < 3; d++) { lo[d] = po[d] / F; hi[d] = (po[d] + pm->shape[d] + F - 1) / F; if (hi[d] > osh[d]) hi[d] = osh[d]; }
    uint8_t *sb = malloc((size_t)S * S * S), *src = nullptr; size_t srccap = 0; int nsh = 0;
    for (int64_t sz = lo[0] / S; sz * S < hi[0]; sz++) for (int64_t sy = lo[1] / S; sy * S < hi[1]; sy++) for (int64_t sx = lo[2] / S; sx * S < hi[2]; sx++) {
        const int64_t so[3] = {sz * S, sy * S, sx * S}, sn[3] = {osh[0] - so[0] < S ? osh[0] - so[0] : S, osh[1] - so[1] < S ? osh[1] - so[1] : S, osh[2] - so[2] < S ? osh[2] - so[2] : S};
        memset(sb, 0, (size_t)S * S * S);
        if (oz) { uint8_t *t = malloc((size_t)sn[0] * sn[1] * sn[2]); if (t && !z3_read(oz, so, sn, t, nt)) for (int64_t a = 0; a < sn[0]; a++) for (int64_t b = 0; b < sn[1]; b++) memcpy(sb + ((size_t)a * S + b) * S, t + ((size_t)a * sn[1] + b) * sn[2], (size_t)sn[2]); free(t); }
        int64_t c0[3], c1[3];   /* this shard's output voxels inside the box */
        for (int d = 0; d < 3; d++) { c0[d] = lo[d] > so[d] ? lo[d] : so[d]; c1[d] = hi[d] < so[d] + sn[d] ? hi[d] : so[d] + sn[d]; }
        if (c1[0] <= c0[0] || c1[1] <= c0[1] || c1[2] <= c0[2]) continue;
        for (int64_t a0 = c0[0]; a0 < c1[0]; a0 += 32) {   /* 32 output rows at a time: a ~0.5 GB source slab (a whole 2048^3 box: 8.7 GB) */
            const int64_t a1 = a0 + 32 < c1[0] ? a0 + 32 : c1[0], cl[3] = {a0, c0[1], c0[2]}, ch[3] = {a1, c1[1], c1[2]};
            int64_t ro[3], rn[3];   /* the source voxels behind them (box-local) */
            for (int d = 0; d < 3; d++) { ro[d] = cl[d] * F - po[d]; int64_t e = ch[d] * F - po[d]; if (ro[d] < 0) ro[d] = 0; if (e > pm->shape[d]) e = pm->shape[d]; rn[d] = e - ro[d]; }
            if (rn[0] <= 0 || rn[1] <= 0 || rn[2] <= 0) continue;
            const size_t need = (size_t)rn[0] * rn[1] * rn[2]; if (need > srccap) { free(src); src = malloc(need); srccap = need; }
            if (!src || z3_read(pz, ro, rn, src, nt)) { fprintf(stderr, "void-pool: %s\n", z3_error()); return 1; }
            for (int64_t a = a0; a < a1; a++) for (int64_t b = c0[1]; b < c1[1]; b++) for (int64_t c = c0[2]; c < c1[2]; c++) {
                int tot = 0, vd = 0;
                for (int64_t za = a * F - po[0]; za < (a + 1) * F - po[0]; za++) { if (za < ro[0] || za >= ro[0] + rn[0]) continue;
                    for (int64_t ya = b * F - po[1]; ya < (b + 1) * F - po[1]; ya++) { if (ya < ro[1] || ya >= ro[1] + rn[1]) continue;
                        for (int64_t xa = c * F - po[2]; xa < (c + 1) * F - po[2]; xa++) { if (xa < ro[2] || xa >= ro[2] + rn[2]) continue;
                            tot++; vd += src[((size_t)(za - ro[0]) * rn[1] + (ya - ro[1])) * rn[2] + (xa - ro[2])] < thr; } } }
                if (tot) sb[((size_t)(a - so[0]) * S + (b - so[1])) * S + (c - so[2])] = (uint8_t)((vd * 255 + tot / 2) / tot);
            }
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, nt)) { fprintf(stderr, "void-pool: %s\n", z3w_error()); return 1; }
        nsh++;
    }
    z3w_close(w); if (oz) z3_close(oz); z3_close(pz); free(sb); free(src);
    fprintf(stderr, "void-pool: %d shards of %s updated from %s\n", nsh, od, argv[2]);
    return 0;
}

/* ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx [--um 2.4] [--local 8] [--local-sigma 1.5] [--depth 30]
   [--margin 0.1] [--sigma 1] [--cut 50,128] [--band 0] [--cutoff T] [--threads 8] [--cache DIR]
   The task papyrus training labels (src/papyrus.h) of a CT box as a box-local uint8 array <out-group>/<um> (ufsm.origin_zyx, so
   gap-compare and the overlays place it): 255 papyrus, 0 void or outside the scroll, 128 ignored. Gap labels by default (--local
   R, papyrus_local); --cutoff T: the old global cutoff at T instead. Per 512^3 output shard the CT is read once with a halo covering
   the smoothing and the closing window, then labelled in 256^3 tiles, one per thread. */
typedef struct { const uint8_t *ct; int64_t cn[3]; const papyrus_cfg *c; float t; int T, H, P; uint8_t *out; int64_t on[3], oo[3]; _Atomic int next; } plab_job;
static void *plab_tiles(void *arg) {
    plab_job *J = arg; const int P = J->P, T = J->T, H = J->H; const size_t p3 = (size_t)P * P * P;
    uint8_t *ct = malloc(p3), *tg = malloc(p3), *ig = malloc(p3), *fg = malloc(p3), *sc = malloc(8 * p3);
    if (!ct || !tg || !ig || !fg || !sc) abort();
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < 8;) {
        const int64_t t0[3] = {(k >> 2 & 1) * (int64_t)T, (k >> 1 & 1) * (int64_t)T, (k & 1) * (int64_t)T};   /* tile origin in the shard */
        if (t0[0] >= J->on[0] || t0[1] >= J->on[1] || t0[2] >= J->on[2]) continue;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {   /* the haloed tile from the shard's CT (zero outside it) */
            const int64_t a = t0[0] + z, b = t0[1] + y; uint8_t *d = ct + ((size_t)z * P + y) * P;
            if (a >= J->cn[0] || b >= J->cn[1]) { memset(d, 0, P); continue; }
            const int64_t nx = J->cn[2] - t0[2] < P ? J->cn[2] - t0[2] : P;
            memcpy(d, J->ct + ((size_t)a * J->cn[1] + b) * J->cn[2] + t0[2], (size_t)nx); if (nx < P) memset(d + nx, 0, (size_t)(P - nx));
        }
        memset(ig, 0, p3);
        papyrus_labels(ct, P, J->c, J->t, 0.0, tg, ig, fg, sc);
        for (int z = 0; z < T && t0[0] + z < J->on[0]; z++) for (int y = 0; y < T && t0[1] + y < J->on[1]; y++) {
            const size_t s = ((size_t)(z + H) * P + y + H) * P + H; uint8_t *o = J->out + ((size_t)(t0[0] + z) * J->on[1] + t0[1] + y) * J->on[2] + t0[2];
            for (int x = 0; x < T && t0[2] + x < J->on[2]; x++) o[x] = ig[s + x] ? 128 : tg[s + x];
        }
    }
    free(ct); free(tg); free(ig); free(fg); free(sc);
    return nullptr;
}
static int cmd_papyrus_labels(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx [--um 2.4] [--local 8] [--local-sigma 1.5] [--depth 30] [--margin 0.1] [--sigma 1] [--cut 50,128] [--band 0] [--cutoff T] [--threads 8] [--cache DIR]\n"); return 2; }
    long long bx[6]; papyrus_cfg c; memset(&c, 0, sizeof c);
    if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", bx, bx + 1, bx + 2, bx + 3, bx + 4, bx + 5) != 6) { fprintf(stderr, "papyrus-labels: --box z,y,x,nz,ny,nx\n"); return 2; }
    const float cutoff = (float)atof(opt(argc, argv, "--cutoff", "-1"));
    c.local = cutoff >= 0 ? 0 : atoi(opt(argc, argv, "--local", "8")); c.local_sigma = (float)atof(opt(argc, argv, "--local-sigma", "1.5"));
    c.local_depth = (float)atof(opt(argc, argv, "--depth", "30")); c.local_margin = (float)atof(opt(argc, argv, "--margin", "0.1"));
    c.sigma = (float)atof(opt(argc, argv, "--sigma", "1")); c.band = (float)atof(opt(argc, argv, "--band", "0"));
    if (sscanf(opt(argc, argv, "--cut", "50,128"), "%f,%f", &c.t_lo, &c.t_hi) != 2 || !(c.sigma >= 0 && c.sigma <= PAPYRUS_RMAX / 3.f) ||
        !(c.local_sigma >= 0 && c.local_sigma <= PAPYRUS_RMAX / 3.f) || (cutoff < 0 && c.local < 1)) { fprintf(stderr, "papyrus-labels: bad label options\n"); return 2; }
    const int nt = atoi(opt(argc, argv, "--threads", "8")), S = 512, T = S / 2;
    float w[2 * PAPYRUS_RMAX + 1]; const int rs = papyrus_taps(c.sigma, w), rl = papyrus_taps(c.local_sigma, w);
    const int H = cutoff >= 0 ? rs : (rs > 2 * c.local + rl ? rs : 2 * c.local + rl), P = T + 2 * H;   /* what a core voxel's label can see */
    z3 *cz = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "papyrus-labels: %s\n", z3_error()); return 1; }
    const z3_meta *cm = z3_meta_of(cz); const double um = atof(opt(argc, argv, "--um", "2.4"));
    const int64_t shp[3] = {bx[3], bx[4], bx[5]};
    char lv[32], od[1400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    char at[600];
    if (cutoff >= 0) snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus labels at cutoff %g (255 papyrus, 0 void, 128 ignored)\",\"origin_zyx\":[%lld,%lld,%lld],\"sigma\":%g,\"band\":%g}}", (double)cutoff, bx[0], bx[1], bx[2], (double)c.sigma, (double)c.band);
    else snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus gap labels (255 papyrus, 0 void, 128 ignored)\",\"origin_zyx\":[%lld,%lld,%lld],\"sigma\":%g,\"cut\":[%g,%g],\"local\":%d,\"local_sigma\":%g,\"local_depth\":%g,\"local_margin\":%g,\"band\":%g}}",
                  bx[0], bx[1], bx[2], (double)c.sigma, (double)c.t_lo, (double)c.t_hi, c.local, (double)c.local_sigma, (double)c.local_depth, (double)c.local_margin, (double)c.band);
    z3w *ow = z3w_create(od, shp, S, 0, 0, at); if (!ow || z3w_write_group(argv[4], &um, 1, "ufsm-papyrus-labels", nullptr)) { fprintf(stderr, "papyrus-labels: %s\n", z3w_error()); return 1; }
    const size_t cap = (size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H);
    uint8_t *cbuf = malloc(cap), *obuf = malloc((size_t)S * S * S); if (!cbuf || !obuf) { fprintf(stderr, "papyrus-labels: out of memory\n"); return 1; }
    const double t0 = now(); int nsh = 0; const int64_t ns[3] = {(shp[0] + S - 1) / S, (shp[1] + S - 1) / S, (shp[2] + S - 1) / S};
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        const int64_t so[3] = {sz * S, sy * S, sx * S}, on[3] = {shp[0] - so[0] < S ? shp[0] - so[0] : S, shp[1] - so[1] < S ? shp[1] - so[1] : S, shp[2] - so[2] < S ? shp[2] - so[2] : S};
        int64_t want0[3], cn[3], ro[3], rn[3];   /* the haloed shard in volume coordinates, and the part inside the volume */
        for (int d = 0; d < 3; d++) { want0[d] = bx[d] + so[d] - H; cn[d] = on[d] + 2 * H; ro[d] = want0[d] < 0 ? 0 : want0[d]; int64_t e = want0[d] + cn[d]; if (e > cm->shape[d]) e = cm->shape[d]; rn[d] = e - ro[d]; }
        memset(cbuf, 0, (size_t)cn[0] * cn[1] * cn[2]);
        if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
            uint8_t *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
            if (!tmp || z3_read(cz, ro, rn, tmp, nt)) { fprintf(stderr, "papyrus-labels: %s\n", z3_error()); return 1; }
            for (int64_t a = 0; a < rn[0]; a++) for (int64_t b = 0; b < rn[1]; b++)
                memcpy(cbuf + ((size_t)(ro[0] - want0[0] + a) * cn[1] + (ro[1] - want0[1] + b)) * cn[2] + (ro[2] - want0[2]), tmp + ((size_t)a * rn[1] + b) * rn[2], (size_t)rn[2]);
            free(tmp);
        }
        memset(obuf, 0, (size_t)S * S * S);
        plab_job J = {cbuf, {cn[0], cn[1], cn[2]}, &c, cutoff, T, H, P, obuf, {on[0], on[1], on[2]}, {0, 0, 0}, 0};
        /* the job writes rows of on[2] voxels; the shard writer wants S^3 */
        uint8_t *dense = malloc((size_t)on[0] * on[1] * on[2]); if (!dense) abort(); J.out = dense;
        pthread_t th[8]; const int k = nt < 1 ? 1 : nt > 8 ? 8 : nt;
        for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, plab_tiles, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
        for (int64_t a = 0; a < on[0]; a++) for (int64_t b = 0; b < on[1]; b++) memcpy(obuf + ((size_t)a * S + b) * S, dense + ((size_t)a * on[1] + b) * on[2], (size_t)on[2]);
        free(dense);
        if (z3w_write_shard(ow, sz, sy, sx, obuf, nt)) { fprintf(stderr, "papyrus-labels: %s\n", z3w_error()); return 1; }
        fprintf(stderr, "\rpapyrus-labels: shard %d / %lld  %.0fs", ++nsh, (long long)(ns[0] * ns[1] * ns[2]), now() - t0);
    }
    z3w_close(ow); z3_close(cz); free(cbuf); free(obuf);
    fprintf(stderr, "\npapyrus-labels: %s (%s labels, halo %d) in %.0fs\n", od, cutoff >= 0 ? "cutoff" : "gap", H, now() - t0);
    return 0;
}

/* An 8-bit RGB PNG (zlib deflate, filter 0 rows) */
static void png_chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    const uint8_t be[4] = {(uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n}; fwrite(be, 1, 4, f); fwrite(type, 1, 4, f);
    if (n) fwrite(d, 1, n, f);
    uLong c = crc32(0L, Z_NULL, 0); c = crc32(c, (const Bytef *)type, 4); if (n) c = crc32(c, d, n);
    const uint8_t cb[4] = {(uint8_t)(c >> 24), (uint8_t)(c >> 16), (uint8_t)(c >> 8), (uint8_t)c}; fwrite(cb, 1, 4, f);
}
static int png_write_rgb(const char *path, int w, int h, const uint8_t *rgb) {
    const size_t rs = (size_t)w * 3 + 1, raw = rs * (size_t)h; uint8_t *r = malloc(raw); uLongf zn = compressBound(raw); uint8_t *z = malloc(zn);
    if (!r || !z) { free(r); free(z); return 1; }
    for (int y = 0; y < h; y++) { r[(size_t)y * rs] = 0; memcpy(r + (size_t)y * rs + 1, rgb + (size_t)y * w * 3, (size_t)w * 3); }
    FILE *f = compress2(z, &zn, r, raw, 6) == Z_OK ? fopen(path, "wb") : nullptr;
    if (f) {
        static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10}; fwrite(sig, 1, 8, f);
        const uint8_t ih[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
        png_chunk(f, "IHDR", ih, 13); png_chunk(f, "IDAT", z, (uint32_t)zn); png_chunk(f, "IEND", nullptr, 0);
    }
    free(r); free(z);
    return f ? fclose(f) != 0 : 1;
}
/* ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w [--labels G1,G2,..] [--um 2.4] [--zoom 1] [--cache DIR]
   One CT plane (volume coordinates) beside the same plane of each label group (papyrus-labels output: key = --um level, placed by
   its ufsm.origin_zyx): void (0 inside the scroll) tinted blue, ignored (128) yellow, papyrus left grey. Panels left to right in
   --labels order after the plain CT, 4 white pixels apart; --zoom N repeats pixels. */
static int cmd_overlay(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w [--labels G1,G2,..] [--prob G1,G2,..] [--um 2.4] [--zoom 1] [--cache DIR]\n"
                                  "  --labels: papyrus label groups (void blue, ignored yellow); --prob: uint8 probability groups (predict output, binary\n"
                                  "  labels), red with alpha p / 255; each group's level is named by --um, its origin_zyx attribute places it\n"); return 2; }
    long long pz[5]; if (sscanf(opt(argc, argv, "--plane", ""), "%lld,%lld,%lld,%lld,%lld", pz, pz + 1, pz + 2, pz + 3, pz + 4) != 5 || pz[3] < 1 || pz[4] < 1) { fprintf(stderr, "overlay: --plane z,y,x,h,w\n"); return 2; }
    const int H = (int)pz[3], W = (int)pz[4], Z = atoi(opt(argc, argv, "--zoom", "1")) < 1 ? 1 : atoi(opt(argc, argv, "--zoom", "1"));
    char lv[32]; z3w_level_name(atof(opt(argc, argv, "--um", "2.4")), lv, sizeof lv);
    char lab[4096]; snprintf(lab, sizeof lab, "%s", opt(argc, argv, "--labels", "")); const char *names[16]; int nl = 0;
    for (char *s = lab, *tok; nl < 16 && (tok = strtok(s, ",")); s = nullptr) names[nl++] = tok;
    char prb[4096]; snprintf(prb, sizeof prb, "%s", opt(argc, argv, "--prob", "")); const int nlab = nl;   /* panels [0, nlab) labels, then probabilities */
    for (char *s = prb, *tok; nl < 16 && (tok = strtok(s, ",")); s = nullptr) names[nl++] = tok;
    const size_t n2 = (size_t)H * W; uint8_t *ct = malloc(n2), *lb = malloc(n2 * (nl ? nl : 1));
    z3 *cz = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!cz || !ct || !lb) { fprintf(stderr, "overlay: %s\n", z3_error()); return 1; }
    const int64_t o[3] = {pz[0], pz[1], pz[2]}, n[3] = {1, H, W};
    if (z3_read(cz, o, n, ct, 8)) { fprintf(stderr, "overlay: %s\n", z3_error()); return 1; }
    for (int i = 0; i < nl; i++) {
        long long po[3] = {0, 0, 0}; char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", names[i], lv); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
        if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
        const char *q = strstr(jb, "\"origin_zyx\":"); if (q) sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2);
        z3 *lz = z3_open(store_open(names[i]), lv, nullptr); if (!lz) { fprintf(stderr, "overlay: %s: %s\n", names[i], z3_error()); return 1; }
        const int64_t lo[3] = {pz[0] - po[0], pz[1] - po[1], pz[2] - po[2]};
        if (z3_read(lz, lo, n, lb + n2 * i, 8)) { fprintf(stderr, "overlay: %s: %s\n", names[i], z3_error()); return 1; }
        z3_close(lz);
    }
    const int gap = 4, OW = (W * Z + gap) * (nl + 1) - gap, OH = H * Z; uint8_t *img = malloc((size_t)OW * OH * 3); if (!img) return 1;
    memset(img, 255, (size_t)OW * OH * 3);
    for (int p = 0; p <= nl; p++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const uint8_t g = ct[(size_t)y * W + x]; int r = g, gg = g, b = g;
        if (p && p - 1 >= nlab) { const int a = lb[n2 * (p - 1) + (size_t)y * W + x];   /* probability: red over the CT */
                 r = (g * (255 - a) + 255 * a) / 255; gg = (g * (255 - a) + 30 * a) / 255; b = (g * (255 - a) + 20 * a) / 255; }
        else if (p) { const uint8_t v = lb[n2 * (p - 1) + (size_t)y * W + x];
                 if (g && v == 0) { r = g / 2; gg = g / 2 + 30; b = g / 2 + 120; } else if (v == 128) { r = g / 2 + 120; gg = g / 2 + 100; b = g / 2; } }
        for (int dy = 0; dy < Z; dy++) for (int dx = 0; dx < Z; dx++) {
            uint8_t *px = img + (((size_t)y * Z + dy) * OW + (size_t)p * (W * Z + gap) + (size_t)x * Z + dx) * 3;
            px[0] = (uint8_t)(r > 255 ? 255 : r); px[1] = (uint8_t)(gg > 255 ? 255 : gg); px[2] = (uint8_t)(b > 255 ? 255 : b);
        }
    }
    const int rc = png_write_rgb(argv[2], OW, OH, img);
    if (rc) fprintf(stderr, "overlay: cannot write %s\n", argv[2]); else fprintf(stderr, "overlay: %s (%d x %d, %d panels)\n", argv[2], OW, OH, nl + 1);
    z3_close(cz); free(ct); free(lb); free(img);
    return rc;
}

typedef struct {   /* ufsm render: one output tile's pixel rows over threads */
    const float *X, *Y, *Z; int w, h; double gs;   /* the vertex grid; output pixels per grid step */
    const uint8_t *vol, *ct; int64_t o[3], n[3]; int depth;
    uint8_t *img, *cimg; int ow, px0, px1, py0, py1; _Atomic int next;
} render_job;
static void *render_rows(void *arg) {
    render_job *J = arg; const int w = J->w, h = J->h;
    for (int py; (py = J->py0 + atomic_fetch_add(&J->next, 1)) < J->py1;) for (int px = J->px0; px < J->px1; px++) {
        const double gu = px / J->gs, gv = py / J->gs;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = gu - c0, fv = gv - r0;
        const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
        if (J->Z[i00] < 0 || J->Z[i01] < 0 || J->Z[i10] < 0 || J->Z[i11] < 0) continue;
        const float *A[3] = {J->Z, J->Y, J->X}; double P[3], du[3], dv[3];
        for (int d = 0; d < 3; d++) {
            const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
            P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
            du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
        }
        double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
        for (int d = 0; d < 3; d++) N[d] /= nl;
        float m = 0;
        for (int t = -J->depth; t <= J->depth; t++) { const float v = trilin(J->vol, J->n, P[0] + t * N[0] - J->o[0], P[1] + t * N[1] - J->o[1], P[2] + t * N[2] - J->o[2]); if (v > m) m = v; }
        J->img[(size_t)py * J->ow + px] = (uint8_t)lroundf(m);
        if (J->cimg) J->cimg[(size_t)py * J->ow + px] = (uint8_t)lroundf(trilin(J->ct, J->n, P[0] - J->o[0], P[1] - J->o[1], P[2] - J->o[2]));
    }
    return nullptr;
}
static int cmd_render(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8] [--tile 512] [--threads 16]\n"); return 2; }
    const int vi = arg_index(argc, argv, "--volume"), ci = arg_index(argc, argv, "--ct");
    if (vi < 0 || vi + 2 >= argc) { fprintf(stderr, "render: --volume ROOT KEY\n"); return 2; }
    const double scale = atof(opt(argc, argv, "--scale", "1")); const int depth = atoi(opt(argc, argv, "--depth", "8"));
    const int tile = atoi(opt(argc, argv, "--tile", "512")), nthreads = atoi(opt(argc, argv, "--threads", "16"));
    const char *cache = opt(argc, argv, "--cache", nullptr);
    z3 *vz = z3_open(store_open(argv[vi + 1]), argv[vi + 2], cache), *cz = ci > 0 && ci + 2 < argc ? z3_open(store_open(argv[ci + 1]), argv[ci + 2], cache) : nullptr;
    if (!vz || (ci > 0 && !cz)) { fprintf(stderr, "render: %s\n", z3_error()); return 1; }
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (system(cmd)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "render: cannot open %s\n", argv[2]); return 1; }
    struct dirent *de; int nw = 0;
    /* the wraps' union box (+- depth): when it fits UFSM_RENDER_MEM GB (default 8) the volume (and CT) are read once and every tile samples
       them (wraps are rings around the axis: per-tile boxes are nearly the whole cross-section, re-decoded per wrap: 127 wraps of a 3.9 cm
       slab took 226 s) */
    uint8_t *rvol = nullptr, *rct = nullptr; int64_t ro[3] = {0, 0, 0}, rn[3] = {0, 0, 0};
    {
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        while ((de = readdir(dd))) {
            if (de->d_name[0] != 'w') continue;
            char p[1400]; int w, h; const char *ax[3] = {"z", "y", "x"};
            for (int d = 0; d < 3; d++) {
                snprintf(p, sizeof p, "%s/%s/%s.tif", argv[2], de->d_name, ax[d]); float *A = read_tif_f32(p, &w, &h); if (!A) continue;
                for (size_t i = 0; i < (size_t)w * h; i++) if (A[i] >= 0) { if (A[i] < lo[d]) lo[d] = A[i]; if (A[i] > hi[d]) hi[d] = A[i]; }
                free(A);
            }
        }
        rewinddir(dd);
        const double budget = atof(getenv("UFSM_RENDER_MEM") ? getenv("UFSM_RENDER_MEM") : "8") * (1u << 30);
        if (lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2]) {
            for (int d = 0; d < 3; d++) { ro[d] = (int64_t)floor(lo[d]) - depth - 2; if (ro[d] < 0) ro[d] = 0; rn[d] = (int64_t)ceil(hi[d]) + depth + 3 - ro[d]; }
            const double nv = (double)rn[0] * rn[1] * rn[2];
            if (nv * (cz ? 2 : 1) <= budget) {
                rvol = malloc((size_t)nv); rct = cz ? malloc((size_t)nv) : nullptr;
                if (!rvol || (cz && !rct) || z3_read(vz, ro, rn, rvol, 32) || (cz && z3_read(cz, ro, rn, rct, 32))) { fprintf(stderr, "render: read failed: %s\n", z3_error()); return 1; }
                fprintf(stderr, "render: the wraps' box %lldx%lldx%lld read once\n", (long long)rn[0], (long long)rn[1], (long long)rn[2]);
            }
        }
    }
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = read_tif_f32(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = read_tif_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = read_tif_f32(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || w < 2 || h < 2) { free(X); free(Y); free(Z); continue; }
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        double step = 1; { FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); const char *q = strstr(b, "\"scale\": ["); if (q) step = 1 / atof(q + 10); } }
        const double gs = step * scale;   /* output pixels per grid step */
        const int ow = (int)((w - 1) * gs) + 1, oh = (int)((h - 1) * gs) + 1;
        uint8_t *img = calloc((size_t)ow * oh, 1), *cimg = cz ? calloc((size_t)ow * oh, 1) : nullptr;
        /* output tiles: each reads only the volume around its own vertices (+- depth), so a wrap of any size renders in bounded memory */
        for (int ty = 0; ty < oh; ty += tile) for (int tx = 0; tx < ow; tx += tile) {
            const int py1 = ty + tile < oh ? ty + tile : oh, px1 = tx + tile < ow ? tx + tile : ow;
            const int r0 = (int)(ty / gs), r1 = (int)fmin(h - 1, ceil(py1 / gs) + 1), c0 = (int)(tx / gs), c1 = (int)fmin(w - 1, ceil(px1 / gs) + 1);
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) {
                const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
                const double v[3] = {Z[i], Y[i], X[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
            }
            if (lo[0] > hi[0]) continue;
            render_job J = {X, Y, Z, w, h, gs, nullptr, nullptr, {0, 0, 0}, {0, 0, 0}, depth, img, cimg, ow, tx, px1, ty, py1, 0};
            if (rvol) {   /* the region read once */
                J.vol = rvol; J.ct = rct; for (int d = 0; d < 3; d++) { J.o[d] = ro[d]; J.n[d] = rn[d]; }
                pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
                for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, render_rows, &J);
                for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
                continue;
            }
            for (int d = 0; d < 3; d++) { J.o[d] = (int64_t)floor(lo[d]) - depth - 2; if (J.o[d] < 0) J.o[d] = 0; J.n[d] = (int64_t)ceil(hi[d]) + depth + 3 - J.o[d]; }
            const size_t nv = (size_t)J.n[0] * J.n[1] * J.n[2];
            if (nv > ((size_t)4 << 30)) { fprintf(stderr, "render: %s tile %d,%d spans %lldx%lldx%lld voxels (a fold?): skipped; use a smaller --tile\n", de->d_name, ty, tx, (long long)J.n[0], (long long)J.n[1], (long long)J.n[2]); continue; }
            uint8_t *vol = malloc(nv), *ct = cz ? malloc(nv) : nullptr;
            if (!vol || (cz && !ct) || z3_read(vz, J.o, J.n, vol, 16) || (cz && z3_read(cz, J.o, J.n, ct, 16))) { fprintf(stderr, "render: read failed: %s\n", z3_error()); return 1; }
            J.vol = vol; J.ct = ct;
            pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
            for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, render_rows, &J);
            for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
            free(vol); free(ct);
        }
        snprintf(p, sizeof p, "%s/%s.tif", argv[3], de->d_name);
        if (tiffw_write(p, ow, oh, 1, img, 1)) { fprintf(stderr, "render: cannot write %s\n", p); return 1; }
        if (cimg) { snprintf(p, sizeof p, "%s/%s_ct.tif", argv[3], de->d_name); if (tiffw_write(p, ow, oh, 1, cimg, 1)) return 1; }
        free(X); free(Y); free(Z); free(img); free(cimg); nw++;
    }
    closedir(dd); free(rvol); free(rct);
    fprintf(stderr, "render: %d wraps written to %s\n", nw, argv[3]);
    return 0;
}

static int cmd_sample(int argc, char **argv) {
    const char *path = argv[2];
    sample_cfg cfg = sample_cfg_default();
    cfg.P = atoi(opt(argc, argv, "--P", "128"));
    cfg.B = atoi(opt(argc, argv, "--B", "2"));
    cfg.seed = (uint64_t)atoll(opt(argc, argv, "--seed", "0"));
    cfg.nworkers = atoi(opt(argc, argv, "--workers", "4"));
    cfg.augment = !flag(argc, argv, "--noaug");
    cfg.shell = atoi(opt(argc, argv, "--shell", "0"));
    cfg.iso = atoi(opt(argc, argv, "--iso", "0")); cfg.iso_void = atoi(opt(argc, argv, "--iso-void", "0")); cfg.iso_pap = atoi(opt(argc, argv, "--iso-pap", "2")); cfg.iso_lift = atoi(opt(argc, argv, "--iso-lift", "0"));
    cfg.dist_sigma = (float)atof(opt(argc, argv, "--dist-head", "0"));
    int nb = atoi(opt(argc, argv, "--n", "4"));
    const char *out = opt(argc, argv, "--out", ".");
    sources *S = sources_load(path);
    if (!S) return 1;
    fprintf(stderr, "%d sources\n", S->n);
    double t0 = now();
    sampler *sp = sampler_start(S, &cfg);
    if (!sp) return 1;
    int P = cfg.P;
    size_t p3 = (size_t)P * P * P;
    /* montage: one row per patch: CT | recto | sheet | mask | radial-y | radial-x (mid z slice) */
    int cols = 6;
    uint8_t *img = malloc((size_t)cols * P * P);
    for (int bi = 0; bi < nb; bi++) {
        batch *b = sampler_next(sp);
        if (!b) { fprintf(stderr, "sampler stopped\n"); break; }
        for (int i = 0; i < cfg.B; i++) {
            const float *x = b->x + (size_t)i * 4 * p3 + (size_t)(P / 2) * P * P;
            for (int c = 0; c < cols; c++)
                for (int y = 0; y < P; y++)
                    for (int xx = 0; xx < P; xx++) {
                        size_t k = (size_t)y * P + xx;
                        double v;
                        if (c == 0) v = 128 + 40 * x[k];
                        else if (c == 1) v = b->t[(size_t)i * NCH * p3 + 0 * p3 + (size_t)(P / 2) * P * P + k];
                        else if (c == 2) v = b->t[(size_t)i * NCH * p3 + 1 * p3 + (size_t)(P / 2) * P * P + k];
                        else if (c == 3) v = 255 * b->m[(size_t)i * p3 + (size_t)(P / 2) * P * P + k];
                        else v = 128 + 127 * x[(size_t)(c - 2) * p3 + k];
                        img[(size_t)y * cols * P + c * P + xx] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                    }
            char fn[1200];
            snprintf(fn, sizeof fn, "%s/patch_%02d_%d.pgm", out, bi, i);
            write_pgm(fn, img, (int64_t)cols * P, P);
            fprintf(stderr, "%s: src %s level %d corner %lld,%lld,%lld w=[%d,%d]\n", fn, S->src[b->src[i]].name, b->level[i],
                    (long long)b->corner[i][0], (long long)b->corner[i][1], (long long)b->corner[i][2], b->w[i * NCH], b->w[i * NCH + 1]);
        }
        sampler_release(sp, b);
    }
    uint64_t prod, rej;
    sampler_stats(sp, &prod, &rej);
    fprintf(stderr, "%llu patches produced, %llu rejected, %.1fs\n", (unsigned long long)prod, (unsigned long long)rej, now() - t0);
    sampler_stop(sp);
    sources_free(S);
    free(img);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "train")) return cmd_train(argc, argv);
    if (argc < 3) return usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "sample")) return cmd_sample(argc, argv);
    if (!strcmp(cmd, "prefetch")) {   /* ufsm prefetch <sources.json> [--levels 1] [--threads 32] [--fraction 1.0] */
        if (argc < 3) { fprintf(stderr, "usage: ufsm prefetch <sources.json> [--levels L] [--threads N] [--fraction F]\n"); return 2; }
        sources *S = sources_load(argv[2]); if (!S) return 1;
        return sources_prefetch(S, atoi(opt(argc, argv, "--levels", "1")), atoi(opt(argc, argv, "--threads", "32")), atof(opt(argc, argv, "--fraction", "1.0")));
    }
    if (!strcmp(cmd, "ingest-labels")) return cmd_ingest_labels(argc, argv);
    if (!strcmp(cmd, "ingest-kaggle")) return cmd_ingest_kaggle(argc, argv);
    if (!strcmp(cmd, "ingest-mesh")) return cmd_ingest_mesh(argc, argv);
    if (!strcmp(cmd, "ls")) return cmd_ls(argc, argv);
    if (!strcmp(cmd, "ingest-zip")) return cmd_ingest_zip(argc, argv);
    if (!strcmp(cmd, "raster")) return cmd_raster(argc, argv);
    if (!strcmp(cmd, "train")) return cmd_train(argc, argv);
    if (!strcmp(cmd, "predict")) return cmd_predict(argc, argv);
    if (!strcmp(cmd, "eval")) return cmd_eval(argc, argv);
    if (!strcmp(cmd, "band")) return cmd_band(argc, argv);
    if (!strcmp(cmd, "fit")) return cmd_fit(argc, argv);
    if (!strcmp(cmd, "phase-raster")) return cmd_phase_raster(argc, argv);
    if (!strcmp(cmd, "wraps")) return cmd_wraps(argc, argv);
    if (!strcmp(cmd, "stitch")) return cmd_stitch(argc, argv);
    if (!strcmp(cmd, "coreline")) return cmd_coreline(argc, argv);
    if (!strcmp(cmd, "render")) return cmd_render(argc, argv);
    if (!strcmp(cmd, "downsample")) return cmd_downsample(argc, argv);
    if (!strcmp(cmd, "side-votes") || !strcmp(cmd, "codes-merge")) return cmd_side_votes(argc, argv);
    if (!strcmp(cmd, "wind-assign")) return cmd_wind_assign(argc, argv);
    if (!strcmp(cmd, "ct-hist")) return cmd_ct_hist(argc, argv);
    if (!strcmp(cmd, "wrapdist")) return cmd_wrapdist(argc, argv);
    if (!strcmp(cmd, "wrapswitch")) return cmd_wrapswitch(argc, argv);
    if (!strcmp(cmd, "surf-snap")) return cmd_surfsnap(argc, argv);
    if (!strcmp(cmd, "refine")) return cmd_refine(argc, argv);
    if (!strcmp(cmd, "surf-stats")) return cmd_surf_stats(argc, argv);
    if (!strcmp(cmd, "fit-compare")) return cmd_fit_compare(argc, argv);
    if (!strcmp(cmd, "field-blend")) return cmd_field_blend(argc, argv);
    if (!strcmp(cmd, "gap-compare")) return cmd_gap_compare(argc, argv);
    if (!strcmp(cmd, "gap-density")) return cmd_gap_density(argc, argv);
    if (!strcmp(cmd, "void-pool")) return cmd_void_pool(argc, argv);
    if (!strcmp(cmd, "papyrus-labels")) return cmd_papyrus_labels(argc, argv);
    if (!strcmp(cmd, "overlay")) return cmd_overlay(argc, argv);
    if (!strcmp(cmd, "wind-score")) return cmd_wind_score(argc, argv);
    if (argc < 4) return usage();
    const char *root = argv[2], *key = argv[3];
    const char *cache = opt(argc, argv, "--cache", nullptr);
    int threads = atoi(opt(argc, argv, "--threads", "0"));
    store *s = store_open(root);
    if (!strcmp(cmd, "regions")) return cmd_regions(s, key);
    if (!strcmp(cmd, "axis")) { if (argc < 5) return usage(); return cmd_axis(s, key, argv[4], cache); }
    if (!strcmp(cmd, "info")) {
        z3_level lv[16];
        int nl = z3_group_levels(s, key, lv, 16);
        if (nl > 0) {
            printf("group %s: %d levels\n", key, nl);
            for (int i = 0; i < nl; i++) {
                char ak[2048];
                snprintf(ak, sizeof ak, "%s/%s", key, lv[i].path);
                z3 *z = z3_open(s, ak, cache);
                if (!z) { printf("  %-10s %.4g um  (unreadable: %s)\n", lv[i].path, lv[i].um, z3_error()); continue; }
                const z3_meta *m = z3_meta_of(z);
                printf("  %-10s %.4g um  shape %lld x %lld x %lld  shard %d chunk %d  %s q=%g%s\n", lv[i].path, lv[i].um,
                       (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->shard[0], m->chunk[0],
                       m->q >= 0 ? "volcomp" : m->raw ? "raw" : "?", m->q, m->zstd ? "+zstd" : "");
                z3_close(z);
            }
            return 0;
        }
        z3 *z = z3_open(s, key, cache);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(z);
        printf("array %s\n  shape %lld x %lld x %lld  shard %d x %d x %d  chunk %d x %d x %d  sep '%c'\n  codec %s q=%g%s  scale %.4g um\n",
               key, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->shard[0], m->shard[1], m->shard[2],
               m->chunk[0], m->chunk[1], m->chunk[2], m->sep, m->q >= 0 ? "volcomp" : m->raw ? "raw" : "?", m->q,
               m->zstd ? "+zstd" : "", m->scale_um);
        z3_close(z);
        return 0;
    }
    if (!strcmp(cmd, "read") || !strcmp(cmd, "slice")) {
        z3 *z = !strcmp(cmd, "read") ? z3_open_typed(s, key, cache) : z3_open(s, key, cache);   /* read: typed arrays as raw elements */
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(z);
        int64_t o[3], n[3];
        const char *out;
        if (!strcmp(cmd, "read")) {
            if (argc < 11) return usage();
            for (int i = 0; i < 3; i++) { o[i] = atoll(argv[4 + i]); n[i] = atoll(argv[7 + i]); }
            out = argv[10];
        } else {
            if (argc < 6) return usage();
            o[0] = atoll(argv[4]); o[1] = o[2] = 0;
            n[0] = 1; n[1] = m->shape[1]; n[2] = m->shape[2];
            out = argv[5];
        }
        size_t nv = (size_t)n[0] * n[1] * n[2]; const int es = m->esize, dt_ = m->dtype;
        uint8_t *buf = malloc(nv * es);
        double t0 = now();
        int rc = z3_read(z, o, n, buf, threads);
        double dt = now() - t0;
        if (rc) { fprintf(stderr, "read failed: %s\n", z3_error()); return 1; }
        size_t nz = 0; double sum = 0;
        for (size_t i = 0; i < nv; i++) {
            double v;
            switch (dt_) {
            case 1: { uint16_t u; memcpy(&u, buf + 2 * i, 2); v = u; break; }
            case 2: { int16_t u; memcpy(&u, buf + 2 * i, 2); v = u; break; }
            case 3: { uint32_t u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            case 4: { int32_t u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            case 5: { float u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            default: v = buf[i];
            }
            nz += v != 0; sum += v;
        }
        fprintf(stderr, "read %zu voxels in %.2fs (%.1f Mvox/s), nonzero %.1f%%, mean %.2f\n", nv, dt, nv / dt / 1e6,
                100.0 * nz / nv, sum / nv);
        if (!strcmp(cmd, "slice")) rc = write_pgm(out, buf, n[2], n[1]);
        else { FILE *f = fopen(out, "wb"); rc = f ? (fwrite(buf, es, nv, f) == nv ? 0 : -1) : -1; if (f) fclose(f); }
        free(buf);
        z3_close(z);
        return rc ? 1 : 0;
    }
    return usage();
}
