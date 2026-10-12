/* ufsm CLI. */
#include "sample.h"
#include "sources.h"
#include "band.h"
#include "edt.h"
#include "store.h"
#include "zarr3.h"
#include "fit.h"
#include "json.h"
#include "nn.h"
#include "tiffw.h"
#include "z3w.h"
#include "tiff.h"
#include <dirent.h>
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
        "  ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase  one array halved per axis (labels, phase)\n"
        "  ufsm fit <fields-dir> [--slab Z] [--out MODEL] [--winding ZARR] [--codes ROOT KEY]   winding fitter (src/fit.h)\n"
        "  ufsm wraps <model-dir> <out-dir> [--step 8] [--fields DIR]          one tifxyz per wrap (level sets of W)\n"
        "  ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY]  ink / CT along each wrap (tiled TIFF)\n"
        "  ufsm wrapdist <wraps-dir> <labels-root> <labels-key>              wrap vertices vs the nearest labelled sheet\n"
        "  ufsm fit-compare <reference-fields> <fields>                      fit fields vs a reference (phase, spacing, normals)\n"
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

/* ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256]
   The band field's winding phase (src/band.h PHASE_*, what the sampler computes per window for --side 2 / task unwrap) of a whole
   winding_mod14 raster, on its label grid: written as <out-dir>/4.8 (uint8, lossless volcomp, fill PHASE_UNKNOWN) plus the group;
   a source then reads it with "band": {..., "phase": {"root": <out-dir>, "key": "4.8"}}. Blocks of --block label voxels plus the
   band halo, in parallel, shard by shard (512 label voxels); blocks without codes stay unknown. --box (native) limits it.
   --pool 2 writes only the 2x circular-mean pooling, <out-dir>/9.6 (a whole scroll at 4.8 um is ~25 MB per shard, 9.6 um
   ~5 MB); --resume 1 skips shards already written (long runs). */
typedef struct { z3 *codes; const int64_t *shape; const axis *ax; band_params bp; int halo, blk; int64_t s0[3]; uint8_t *shard; atomic_int next; int nblk[3]; size_t nonempty; pthread_mutex_t mu; } praster_job;
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
        if (!codes || !tmp || z3_read(J->codes, ro, rn, tmp, 1)) { free(codes); free(tmp); fprintf(stderr, "phase-raster: read failed: %s\n", z3_error()); continue; }
        size_t nz = 0; for (size_t k = 0; k < (size_t)rn[0] * rn[1] * rn[2]; k++) nz += tmp[k] != 0;
        if (!nz) { free(codes); free(tmp); continue; }
        for (int64_t zz = 0; zz < rn[0]; zz++) for (int64_t yy = 0; yy < rn[1]; yy++)
            memcpy(codes + ((size_t)(zz + ro[0] - o[0]) * n[1] + (yy + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), tmp + ((size_t)zz * rn[1] + yy) * rn[2], (size_t)rn[2]);
        free(tmp);
        double *cy = malloc(n[0] * sizeof(double)), *cx = malloc(n[0] * sizeof(double));
        for (int64_t zz = 0; zz < n[0]; zz++) axis_at(J->ax, 2.0 * (zz + o[0]) + 0.5, &cy[zz], &cx[zz]);
        uint8_t *band = malloc(N), *phase = malloc(N);
        if (!band || !phase || band_field_phase(codes, ni, o, cy, cx, J->bp, band, phase, nullptr)) { fprintf(stderr, "phase-raster: out of memory\n"); free(codes); free(band); free(phase); free(cy); free(cx); continue; }
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
static int cmd_phase_raster(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256] [--pool 2] [--resume 1]\n"); return 2; }
    const int pool = atoi(opt(argc, argv, "--pool", "1")), resume = atoi(opt(argc, argv, "--resume", "0"));   /* --pool 2: write only the 9.6 um (2x circular-mean) phase; --resume: skip written shards */
    if (pool != 1 && pool != 2) { fprintf(stderr, "phase-raster: --pool 1|2\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75"))};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "phase-raster: cannot load axis\n"); return 1; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int64_t shape[3] = {m->shape[0], m->shape[1], m->shape[2]};
    const int blk = atoi(opt(argc, argv, "--block", "256")), nth = atoi(opt(argc, argv, "--threads", "4"));
    if (512 % blk) { fprintf(stderr, "phase-raster: --block must divide 512\n"); return 2; }
    int64_t lo[3] = {0, 0, 0}, hi[3] = {shape[0], shape[1], shape[2]};
    const char *bx = opt(argc, argv, "--box", nullptr);
    if (bx) { long long v[6]; if (sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return usage(); for (int d = 0; d < 3; d++) { lo[d] = v[d] / 2; hi[d] = (v[d] + v[3 + d] + 1) / 2; } }
    const double um = 4.8 * pool; char lv[32]; z3w_level_name(um, lv, sizeof lv);
    char dir[1400]; snprintf(dir, sizeof dir, "%s/%s", argv[4], lv);
    char attrs[600]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown)%s\",\"codes\":\"%s/%s\",\"radius\":%g,\"span\":%g}}",
                                   pool == 2 ? ", pooled 2x (circular mean of the known codes)" : "", argv[2], argv[3], bp.radius, bp.span);
    const int64_t oshape[3] = {(shape[0] + pool - 1) / pool, (shape[1] + pool - 1) / pool, (shape[2] + pool - 1) / pool};
    z3w *w = z3w_create(dir, oshape, 512 / pool, 0, PHASE_UNKNOWN, attrs);   /* a 4.8 shard of 512 -> one output shard */
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    if (z3w_write_group(argv[4], &um, 1, "ufsm-phase", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    uint8_t *shard = malloc((size_t)512 * 512 * 512), *pooled = pool == 2 ? malloc((size_t)256 * 256 * 256) : nullptr;
    size_t skipped = 0;
    const double t0 = now(); size_t nonempty = 0, nshards = 0;
    for (int64_t sz = lo[0] / 512; sz * 512 < hi[0]; sz++) for (int64_t sy = lo[1] / 512; sy * 512 < hi[1]; sy++) for (int64_t sx = lo[2] / 512; sx * 512 < hi[2]; sx++) {
        if (resume) { char sp[1600]; snprintf(sp, sizeof sp, "%s/c/%lld/%lld/%lld", dir, (long long)sz, (long long)sy, (long long)sx); if (access(sp, F_OK) == 0) { skipped++; continue; } }
        memset(shard, PHASE_UNKNOWN, (size_t)512 * 512 * 512);
        praster_job J = {z, shape, &ax, bp, (int)ceilf((bp.radius + bp.span) / 2.f) + 2, blk, {sz * 512, sy * 512, sx * 512}, shard, 0, {512 / blk, 512 / blk, 512 / blk}, 0, PTHREAD_MUTEX_INITIALIZER};
        pthread_t th[64]; const int nt = nth < 1 ? 1 : nth > 64 ? 64 : nth;
        for (int k = 0; k < nt; k++) pthread_create(&th[k], nullptr, praster_worker, &J);
        for (int k = 0; k < nt; k++) pthread_join(th[k], nullptr);
        if (J.nonempty && pooled) phase_pool2(shard, pooled);
        if (J.nonempty && z3w_write_shard(w, sz, sy, sx, pooled ? pooled : shard, nt)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        nonempty += J.nonempty; nshards++;
        fprintf(stderr, "\rshard %lld,%lld,%lld: %zu blocks with codes so far, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, nonempty, now() - t0);
    }
    if (z3w_close(w)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    fprintf(stderr, "\nphase-raster: %zu shards (%zu already written), %zu blocks with codes, %.0fs -> %s\n", nshards, skipped, nonempty, now() - t0, dir);
    free(shard); free(pooled); z3_close(z);
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
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75"))};
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
        uint8_t *tmp = malloc((size_t)rn2[0] * rn2[1] * rn2[2]);
        if (z3_read(pz, ro2, rn2, tmp, 8)) { fprintf(stderr, "band: phase read failed: %s\n", z3_error()); return 1; }
        for (int64_t zz = 0; zz < rn2[0]; zz++) for (int64_t yy = 0; yy < rn2[1]; yy++)
            memcpy(phs + ((size_t)(zz + ro2[0] - po[0]) * pnn[1] + (yy + ro2[1] - po[1])) * pnn[2] + (ro2[2] - po[2]), tmp + ((size_t)zz * rn2[1] + yy) * rn2[2], (size_t)rn2[2]);
        free(tmp); z3_close(pz);
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
   [--model MODEL-DIR]
   The winding fitter (src/fit.h) on a directory of fit fields (ufsm band --fit-fields, predict --fit-fields / --fit-zarr). --slab:
   a global coarse fit plus z-slabs of Z native voxels (a fit set; --out writes it, --model reads a set or a model). --codes: score
   against a winding_mod14 band raster (label grid, 2 native voxels per label voxel, code = 1 + round(18 q) mod 252 with q the mesh
   coordinate, q = U for dir +1): on every labelled voxel of the box U_fit - q must be one constant mod 14 turns. */
static int cmd_fit(int argc, char **argv) {
    const char *dir = argv[2];
    fit_cfg c = fit_cfg_default();
    c.verbose = arg_index(argc, argv, "--quiet") < 0;
    c.dir = atoi(opt(argc, argv, "--dir", "0")); c.unwrap_voxels = atoi(opt(argc, argv, "--unwrap-voxels", "1"));
    c.analytic = atoi(opt(argc, argv, "--analytic", "1")); c.packed_um = atof(opt(argc, argv, "--packed-um", "0")); c.batch = atoi(opt(argc, argv, "--batch", "2000000")); c.knot_dz = atoi(opt(argc, argv, "--knot-dz", "1024")); c.phase_stage = atoi(opt(argc, argv, "--phase-stage", "4"));
    if (*opt(argc, argv, "--lattice", "")) { c.nstage = 0; char *t = strdup(opt(argc, argv, "--lattice", "")); for (char *q = strtok(t, ","); q && c.nstage < 8; q = strtok(nullptr, ",")) c.lattice[c.nstage++] = atoi(q); free(t); }
    if (*opt(argc, argv, "--steps", "")) { int k = 0; char *t = strdup(opt(argc, argv, "--steps", "")); for (char *q = strtok(t, ","); q && k < 8; q = strtok(nullptr, ",")) c.steps[k++] = atoi(q); for (; k < c.nstage; k++) c.steps[k] = c.steps[k - 1]; free(t); }
    nn_init(atoi(opt(argc, argv, "--gpu", "0")));
    const double t0 = now();
    const char *min = opt(argc, argv, "--model", nullptr);   /* score / export a saved model or set instead of fitting */
    const int slab = atoi(opt(argc, argv, "--slab", "0")), overlap = atoi(opt(argc, argv, "--overlap", "256"));
    fit_fields_buf fb; memset(&fb, 0, sizeof fb);
    if (fit_fields_load(dir, 0, 0, min || slab > 0 ? 0 : 1, &fb)) { fprintf(stderr, "fit: cannot load the fields in %s\n", dir); return 1; }   /* (slab runs load per slab; a given model needs the geometry only) */
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
        fit_set_dev *gd = fit_set_upload(fs); if (!gd) return 1;
        for (int64_t sz = 0; sz * 128 < gs[0]; sz++) for (int64_t sy = 0; sy * 128 < gs[1]; sy++) for (int64_t sx = 0; sx * 128 < gs[2]; sx++) {
            const int64_t n3[3] = {gs[0] - sz * 128 < 128 ? gs[0] - sz * 128 : 128, gs[1] - sy * 128 < 128 ? gs[1] - sy * 128 : 128, gs[2] - sx * 128 < 128 ? gs[2] - sx * 128 : 128};
            const double o3[3] = {(double)sz * 128 * ws, (double)sy * 128 * ws, (double)sx * 128 * ws};
            if (fit_set_grid(gd, fs, o3, n3, ws, 0, wv)) { fprintf(stderr, "fit: no device memory for the winding grid\n"); return 1; }
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
        store *s = store_open(argv[ci + 1]); z3 *z = z3_open(s, argv[ci + 2], nullptr);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const double cf = atof(argv[ci + 2]) > 0 ? atof(argv[ci + 2]) / fb.um : 2;   /* label voxel in native voxels (the codes' key is its um: 4.8) */
        int64_t lo[3], ln[3]; for (int d = 0; d < 3; d++) { lo[d] = (int64_t)floor(org[d] / cf); ln[d] = (int64_t)(shp[d] * h / cf); }
        enum { NB = 14 * 72, NF = 14 * 1024 }; size_t *hist = calloc(NB, sizeof(size_t)), *fine = calloc(NF, sizeof(size_t)), nd = 0;   /* d = U - q mod 14: 1/72 and 1/1024 turn bins */
        const int detail = arg_index(argc, argv, "--score-detail") > 0;   /* integer errors by z octile and by radius */
        typedef struct { float d; uint8_t z, r; } score_rec; score_rec *dd = nullptr; size_t ndet = 0, dcap = 0;
        const int na_ax = (int)(shp[0] * h) + 1;
        FILE *dump = *opt(argc, argv, "--score-dump", "") ? fopen(opt(argc, argv, "--score-dump", ""), "wb") : nullptr;   /* box-local z y x, U - q mod 14 per voxel */
        const size_t plane = (size_t)ln[1] * ln[2]; int64_t zc = (int64_t)((32u << 20) / plane); if (zc < 1) zc = 1;
        uint8_t *codes = malloc(zc * plane); float *U = malloc(zc * plane * 4);
        fit_set_dev *gd = fit_set_upload(fs); if (!gd) return 1;
        for (int64_t z0 = 0; z0 < ln[0]; z0 += zc) {
            const int64_t nz = ln[0] - z0 < zc ? ln[0] - z0 : zc;
            const int64_t o3[3] = {lo[0] + z0, lo[1], lo[2]}, n3[3] = {nz, ln[1], ln[2]};
            if (z3_read(z, o3, n3, codes, 8)) { fprintf(stderr, "fit: code read failed: %s\n", z3_error()); return 1; }
            size_t any = 0; for (size_t i = 0; i < (size_t)nz * plane && !any; i++) any = codes[i] >= 1 && codes[i] <= 252;
            if (!any) continue;
            const double og[3] = {cf * o3[0] - 0.25 * cf - org[0], cf * lo[1] - 0.25 * cf - org[1], cf * lo[2] - 0.25 * cf - org[2]};   /* label voxel i at cf (i + lo + 1 / 4) - org */
            if (fit_set_grid(gd, fs, og, n3, cf, 1, U)) { fprintf(stderr, "fit: no device memory for the score grid\n"); return 1; }
            for (int64_t zz = 0; zz < nz; zz++) for (int64_t yy = 0; yy < ln[1]; yy++) for (int64_t xx = 0; xx < ln[2]; xx++) {
                const size_t i = ((size_t)zz * ln[1] + yy) * ln[2] + xx; const uint8_t cd = codes[i];
                if (cd < 1 || cd > 252) continue;
                const double pz = cf * (zz + o3[0] + 0.25) - org[0], py = cf * (yy + lo[1] + 0.25) - org[1], px = cf * (xx + lo[2] + 0.25) - org[2];
                if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
                double d = fmod(U[i] - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14; if (d >= 14) d = 0;
                if (dump) { const float rec[4] = {(float)pz, (float)py, (float)px, (float)d}; if (fwrite(rec, 4, 4, dump) != 4) { fclose(dump); dump = nullptr; } }
                hist[(int)(d / 14 * NB) % NB]++; fine[(int)(d / 14 * NF) % NF]++; nd++;
                if (detail) {   /* d and the voxel's z / radius bins for the breakdown after the gauge is known */
                    if (ndet == dcap) { dcap = dcap ? 2 * dcap : 1 << 20; dd = realloc(dd, dcap * sizeof *dd); }
                    const int zi = (int)fmin(7, fmax(0, pz / (shp[0] * h) * 8));
                    const int ai = (int)fmin(na_ax - 1, fmax(0, pz)); const double rr = hypot(py - fb.cy[ai], px - fb.cx[ai]) * fb.um;
                    dd[ndet++] = (score_rec){(float)d, (uint8_t)zi, (uint8_t)fmin(255, rr / 250)};   /* radius in 250 um bins */
                }
            }
        }
        fit_set_dev_free(gd); free(codes); free(U);
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
        free(dd);
        free(ah); free(hist); free(fine);
        z3_close(z); if (dump) fclose(dump);
    }
    fit_set_free(fs); fit_fields_free(&fb);
    return 0;
}

/* ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16]
   [--snap ROOT KEY [--snap-range 0.35] [--snap-min 0.5] [--snap-origin z,y,x]] [--out-um U]
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
                 double range, pmin; _Atomic int next; } snap_job;
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
        float best = -1; double bt = 0;
        for (double t = -R; t <= R; t += 0.5) {
            const float v = trilin(J->vol, J->n, J->pz[i] - J->o[0], J->py[i] + t * uy - J->o[1], J->px[i] + t * ux - J->o[2]);
            if (v > best) { best = v; bt = t; }
        }
        if (best >= J->pmin * 255) J->disp[i] = (float)bt;
    }
    return nullptr;
}
static int cmp_fl(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static size_t wraps_snap(z3 *rz, const int64_t ro[3], const fit_model *m, int rows, int cols, float *px, float *py, float *pz, double range, double pmin, int nthreads) {
    const size_t nv = (size_t)rows * cols; float *disp = malloc(nv * 4); for (size_t i = 0; i < nv; i++) disp[i] = NAN;
    const z3_meta *zm = z3_meta_of(rz); double pmax = 0; for (int k = 0; k < m->nk; k++) if (m->pitch[k] > pmax) pmax = m->pitch[k];
    const int TR = 64, TC = 512; const double mar = range * pmax + 3;
    for (int r0 = 0; r0 < rows; r0 += TR) for (int c0 = 0; c0 < cols; c0 += TC) {
        const int r1 = r0 + TR < rows ? r0 + TR : rows, c1 = c0 + TC < cols ? c0 + TC : cols;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
            const size_t i = (size_t)r * cols + c; if (pz[i] < 0) continue;
            const double v[3] = {pz[i], py[i], px[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
        }
        if (lo[0] > hi[0]) continue;
        snap_job J = {m, nullptr, {0, 0, 0}, {0, 0, 0}, px, py, pz, disp, cols, r0, r1, c0, c1, range, pmin, 0};
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
typedef struct {   /* ufsm wraps: one wrap's rows over threads */
    const fit_set *fs; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh;
    double step, Z; int na; double th0, dth; int rows, cols, k; double rk;
    float *px, *py, *pz; _Atomic int next; _Atomic size_t valid;
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
            if (fit_set_ray_root(J->fs, z, th, J->k, rprev, &rr)) continue;
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
static int cmd_wraps(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16]\n"); return 2; }
    fit_set *fs = fit_set_load(argv[2]); if (!fs) { fprintf(stderr, "wraps: cannot load %s\n", argv[2]); return 1; }
    const fit_model *m = fs->global;   /* the box, the axis; rows dispatch to the slab model of their z */
    const double step = atof(opt(argc, argv, "--step", "8")), um = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : fs->global->um;
    const int nthreads = atoi(opt(argc, argv, "--threads", "16"));
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
    int nwr = 0; size_t nvalid_all = 0;
    for (int k = k0; k <= k1; k++) {
        /* the wrap's radius at the box centre: the root along the central ray */
        const double zc = Z / 2, cyc = m->cy[(int)fmin(zc, na - 1)], cxc = m->cx[(int)fmin(zc, na - 1)];
        const double thc = atan2(Y / 2 - cyc, X / 2 - cxc); double rk;
        if (fit_set_ray_root(fs, zc, thc, k, hypot(Y / 2 - cyc, X / 2 - cxc), &rk)) rk = hypot(Y / 2 - cyc, X / 2 - cxc);
        const double dth = step / fmax(rk, step);
        const int cols = (int)ceil((th1 - th0) / dth) + 1;
        const size_t nv = (size_t)rows * cols;
        float *px = malloc(nv * 4), *py = malloc(nv * 4), *pz = malloc(nv * 4);
        wraps_job J = {fs, m, wf, {wsh[0], wsh[1], wsh[2]}, wh, step, Z, na, th0, dth, rows, cols, k, rk, px, py, pz, 0, 0};
        pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wraps_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        const size_t valid = atomic_load(&J.valid);
        if (snapz && valid) { const size_t mv = wraps_snap(snapz, snapo, m, rows, cols, px, py, pz, snap_range, snap_min, nthreads); if (verbose_snap) fprintf(stderr, "wraps: w%03d: %zu of %zu vertices snapped\n", k, mv, valid); }
        if (valid > 16) {
            char wd[1400]; snprintf(wd, sizeof wd, "%s/w%03d", argv[3], k);
            double wstep = step, wum = um;
            if (out_um > 0 && fabs(out_um - um) > 1e-9) {   /* --out-um: coordinates on another grid of the same scan (voxel centres to voxel centres) */
                const double f = um / out_um;
                for (size_t i = 0; i < nv; i++) if (pz[i] >= 0) { px[i] = (float)((px[i] + 0.5) * f - 0.5); py[i] = (float)((py[i] + 0.5) * f - 0.5); pz[i] = (float)((pz[i] + 0.5) * f - 0.5); }
                wstep = step * f; wum = out_um;
            }
            if (tifxyz_write(wd, cols, rows, px, py, pz, wstep, wum, k, "ufsm wraps (fitted winding level set)")) { fprintf(stderr, "wraps: cannot write %s\n", wd); return 1; }
            nwr++; nvalid_all += valid;
        }
        free(px); free(py); free(pz);
    }
    fprintf(stderr, "wraps: %d wraps (%d..%d) with %zu vertices written to %s\n", nwr, k0, k1, nvalid_all, argv[3]);
    fit_set_free(fs); free(wf); if (snapz) z3_close(snapz);
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
static int cmd_downsample(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase [--box z,y,x,nz,ny,nx] [--shard 512] [--cache DIR] [--threads 16]\n"); return 2; }
    const char *mode = opt(argc, argv, "--mode", "max"); const int phase = !strcmp(mode, "phase");
    if (!phase && strcmp(mode, "max")) { fprintf(stderr, "downsample: --mode max|phase\n"); return 2; }
    const int S = atoi(opt(argc, argv, "--shard", "512")), nt = atoi(opt(argc, argv, "--threads", "16")); const double um = atof(argv[5]);
    z3 *z = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int fill = phase ? PHASE_UNKNOWN : m->fill;
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    if (*opt(argc, argv, "--box", "")) { long long v[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "downsample: --box z,y,x,nz,ny,nx\n"); return 2; } for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } }
    const int64_t osh[3] = {(m->shape[0] + 1) / 2, (m->shape[1] + 1) / 2, (m->shape[2] + 1) / 2};
    char lv[32], od[1400], at[400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"%s halved (%s)\"%s}}", argv[3], mode, m->label_binary ? ",\"encoding\":\"binary\"" : "");
    z3w *w = z3w_create(od, osh, S, 0, fill, at);
    double lum[16]; int nl = 0;   /* the group lists every level already in it (each a number-named array), sorted, plus this one */
    { DIR *gd = opendir(argv[4]); struct dirent *ge;
      while (gd && (ge = readdir(gd)) && nl < 15) { char *e; const double v = strtod(ge->d_name, &e); char zj[1400]; snprintf(zj, sizeof zj, "%s/%s/zarr.json", argv[4], ge->d_name);
          if (v > 0 && !*e && fabs(v - um) > 1e-6 && access(zj, R_OK) == 0) lum[nl++] = v; }
      if (gd) closedir(gd); }
    lum[nl++] = um;
    for (int a = 1; a < nl; a++) for (int b = a; b > 0 && lum[b] < lum[b - 1]; b--) { const double t = lum[b]; lum[b] = lum[b - 1]; lum[b - 1] = t; }
    if (!w || z3w_write_group(argv[4], lum, nl, "ufsm-downsample", nullptr)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
    uint8_t *sb = malloc((size_t)S * S * S), *src = malloc((size_t)2 * 2 * S * 2 * S);   /* one output shard; two source rows */
    int nsh = 0;
    for (int64_t sz = (bo[0] / 2) / S; sz * S < osh[0] && sz * S * 2 < bo[0] + bn[0]; sz++)
    for (int64_t sy = (bo[1] / 2) / S; sy * S < osh[1] && sy * S * 2 < bo[1] + bn[1]; sy++)
    for (int64_t sx = (bo[2] / 2) / S; sx * S < osh[2] && sx * S * 2 < bo[2] + bn[2]; sx++) {
        memset(sb, fill, (size_t)S * S * S);
        const int64_t ny = osh[1] - sy * S < S ? osh[1] - sy * S : S, nx = osh[2] - sx * S < S ? osh[2] - sx * S : S;
        for (int64_t a = 0; a < S && sz * S + a < osh[0]; a++) {   /* output row a from source rows 2 a, 2 a + 1 */
            const int64_t o[3] = {2 * (sz * S + a), 2 * sy * S, 2 * sx * S};
            const int64_t n[3] = {m->shape[0] - o[0] < 2 ? m->shape[0] - o[0] : 2, m->shape[1] - o[1] < 2 * ny ? m->shape[1] - o[1] : 2 * ny, m->shape[2] - o[2] < 2 * nx ? m->shape[2] - o[2] : 2 * nx};
            if (z3_read(z, o, n, src, nt)) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
            for (int64_t b = 0; b < ny; b++) for (int64_t c = 0; c < nx; c++) {
                double cs = 0, sn = 0; int known = 0, mx = 0;
                for (int k = 0; k < 8; k++) {
                    const int64_t zz = k >> 2 & 1, yy = 2 * b + (k >> 1 & 1), xx = 2 * c + (k & 1);
                    if (zz >= n[0] || yy >= n[1] || xx >= n[2]) continue;
                    const int v = src[((size_t)zz * n[1] + yy) * n[2] + xx];
                    if (!phase) { if (v > mx) mx = v; continue; }
                    if (v >= PHASE_PERIOD) continue;
                    cs += cos(2 * M_PI * v / PHASE_PERIOD); sn += sin(2 * M_PI * v / PHASE_PERIOD); known++;
                }
                uint8_t out = (uint8_t)mx;
                if (phase) {
                    out = PHASE_UNKNOWN;
                    if (known && hypot(cs, sn) >= 0.5 * known) { double t = atan2(sn, cs) / (2 * M_PI); if (t < 0) t += 1; out = (uint8_t)((int)lround(t * PHASE_PERIOD) % PHASE_PERIOD); }
                }
                sb[((size_t)a * S + b) * S + c] = out;
            }
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, nt)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
        nsh++;
    }
    z3w_close(w); free(sb); free(src); z3_close(z);
    fprintf(stderr, "downsample: %d shards of %s written to %s\n", nsh, lv, od);
    return 0;
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
    for (int d = 0; d < 3; d++) {
        if (A.shape[d] != B.shape[d] || fabs(A.origin[d] * A.um - B.origin[d] * B.um) > 0.5 * A.h * A.um || fabs(A.h * A.um - B.h * B.um) > 1e-6 * A.h * A.um) {
            fprintf(stderr, "fit-compare: the field sets cover different cells (shape %d/%d, origin %.1f/%.1f um, cell %.2f/%.2f um on axis %d)\n", A.shape[d], B.shape[d], A.origin[d] * A.um, B.origin[d] * B.um, A.h * A.um, B.h * B.um, d); return 2;
        }
    }
    const size_t n = (size_t)A.shape[0] * A.shape[1] * A.shape[2];
    enum { NP = 500, NC = 100, NR = 200, NA = 900, NS = 5, NK = 4 };
    size_t hp[NP] = {0}, hca[NC] = {0}, hcb[NC] = {0}, hr[NR] = {0}, ha[NA] = {0}, sn[NS] = {0}, sg[NS] = {0}, kn[NK] = {0}, kg[NK] = {0}, m = 0, lt01 = 0, gt25 = 0;
    const double sb[NS + 1] = {0, 100, 140, 180, 240, 1e30}, cb[NK + 1] = {0, 0.2, 0.4, 0.6, 1.01};
    for (size_t i = 0; i < n; i++) {
        if (!A.f.w[i] || !B.f.w[i]) continue;
        const double ac = (A.f.pc[i] - 128) / 127.0, as = (A.f.ps[i] - 128) / 127.0, bc = (B.f.pc[i] - 128) / 127.0, bs = (B.f.ps[i] - 128) / 127.0;
        const double ca = hypot(ac, as), cbv = hypot(bc, bs); if (ca <= 0.3) continue;
        m++;
        double e = (atan2(bs, bc) - atan2(as, ac)) / (2 * M_PI); e -= round(e); e = fabs(e);
        hp[(int)fmin(NP - 1, e / 0.5 * NP)]++; lt01 += e < 0.1; gt25 += e > 0.25;
        hca[(int)fmin(NC - 1, ca * NC)]++; hcb[(int)fmin(NC - 1, cbv * NC)]++;
        const double spa = A.f.sp[i] * A.h / 8.0 * A.um, spb = B.f.sp[i] * B.h / 8.0 * B.um;   /* spacing in um (sp = 8 / (h rho)) */
        if (spa > 0 && spb > 0) hr[(int)fmin(NR - 1, fmax(0, (log2(spb / spa) + 2) / 4 * NR))]++;   /* log2 ratio in [-2, 2) */
        for (int k = 0; k < NS; k++) if (spa >= sb[k] && spa < sb[k + 1]) { sn[k]++; sg[k] += e > 0.25; }
        for (int k = 0; k < NK; k++) if (cbv >= cb[k] && cbv < cb[k + 1]) { kn[k]++; kg[k] += e > 0.25; }
        double va[3], vb[3]; fc_axis(&A.f, i, va); fc_axis(&B.f, i, vb);
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
    closedir(dd);
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
    if (!strcmp(cmd, "render")) return cmd_render(argc, argv);
    if (!strcmp(cmd, "downsample")) return cmd_downsample(argc, argv);
    if (!strcmp(cmd, "wrapdist")) return cmd_wrapdist(argc, argv);
    if (!strcmp(cmd, "fit-compare")) return cmd_fit_compare(argc, argv);
    if (!strcmp(cmd, "gap-compare")) return cmd_gap_compare(argc, argv);
    if (!strcmp(cmd, "gap-density")) return cmd_gap_density(argc, argv);
    if (!strcmp(cmd, "void-pool")) return cmd_void_pool(argc, argv);
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
