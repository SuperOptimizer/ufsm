/* ufsm CLI. */
#include "sample.h"
#include "sources.h"
#include "band.h"
#include "store.h"
#include "zarr3.h"
#include "fit.h"
#include "json.h"
#include "nn.h"
#include "tiffw.h"
#include "z3w.h"
#include "tiff.h"
#include <dirent.h>
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
    const int rc = fit_fields_save(dir, org, shp, h, a, cy, cx, na);
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx);
    return rc;
}

/* ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256]
   The band field's winding phase (src/band.h PHASE_*, what the sampler computes per window for --side 2 / task unwrap) of a whole
   winding_mod14 raster, on its label grid: written as <out-dir>/4.8 (uint8, lossless volcomp, fill PHASE_UNKNOWN) plus the group;
   a source then reads it with "band": {..., "phase": {"root": <out-dir>, "key": "4.8"}}. Blocks of --block label voxels plus the
   band halo, in parallel, shard by shard (512 label voxels); blocks without codes stay unknown. --box (native) limits it. */
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
static int cmd_phase_raster(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256]\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75"))};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "phase-raster: cannot load axis\n"); return 1; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int64_t shape[3] = {m->shape[0], m->shape[1], m->shape[2]};
    const int blk = atoi(opt(argc, argv, "--block", "256")), nth = atoi(opt(argc, argv, "--threads", "4"));
    if (512 % blk) { fprintf(stderr, "phase-raster: --block must divide 512\n"); return 2; }
    int64_t lo[3] = {0, 0, 0}, hi[3] = {shape[0], shape[1], shape[2]};
    const char *bx = opt(argc, argv, "--box", nullptr);
    if (bx) { long long v[6]; if (sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return usage(); for (int d = 0; d < 3; d++) { lo[d] = v[d] / 2; hi[d] = (v[d] + v[3 + d] + 1) / 2; } }
    char dir[1400]; snprintf(dir, sizeof dir, "%s/4.8", argv[4]);
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown)\",\"codes\":\"%s/%s\",\"radius\":%g,\"span\":%g}}", argv[2], argv[3], bp.radius, bp.span);
    z3w *w = z3w_create(dir, shape, 512, 0, PHASE_UNKNOWN, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    const double um = 4.8; if (z3w_write_group(argv[4], &um, 1, "ufsm-phase", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    uint8_t *shard = malloc((size_t)512 * 512 * 512);
    const double t0 = now(); size_t nonempty = 0, nshards = 0;
    for (int64_t sz = lo[0] / 512; sz * 512 < hi[0]; sz++) for (int64_t sy = lo[1] / 512; sy * 512 < hi[1]; sy++) for (int64_t sx = lo[2] / 512; sx * 512 < hi[2]; sx++) {
        memset(shard, PHASE_UNKNOWN, (size_t)512 * 512 * 512);
        praster_job J = {z, shape, &ax, bp, (int)ceilf((bp.radius + bp.span) / 2.f) + 2, blk, {sz * 512, sy * 512, sx * 512}, shard, 0, {512 / blk, 512 / blk, 512 / blk}, 0, PTHREAD_MUTEX_INITIALIZER};
        pthread_t th[64]; const int nt = nth < 1 ? 1 : nth > 64 ? 64 : nth;
        for (int k = 0; k < nt; k++) pthread_create(&th[k], nullptr, praster_worker, &J);
        for (int k = 0; k < nt; k++) pthread_join(th[k], nullptr);
        if (J.nonempty && z3w_write_shard(w, sz, sy, sx, shard, nt)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        nonempty += J.nonempty; nshards++;
        fprintf(stderr, "\rshard %lld,%lld,%lld: %zu blocks with codes so far, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, nonempty, now() - t0);
    }
    if (z3w_close(w)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    fprintf(stderr, "\nphase-raster: %zu shards, %zu blocks with codes, %.0fs -> %s\n", nshards, nonempty, now() - t0, argv[4]);
    free(shard); z3_close(z);
    return 0;
}

/* ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis umbilicus.json] [--radius 80] [--span 75] [--threads 8] [--side side.raw]
   [--phase phase.raw]
   Band field (src/band.h) of a native-voxel box (even origin and size) from a winding_mod14 raster level (label grid, 2 native
   voxels per label voxel): writes the band per label voxel, (nz/2) x (ny/2) x (nx/2) uint8, 0..251 or 255 unknown. */
static int cmd_band(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis A] [--radius 80] [--span 75]\n"); return 2; }
    long long b[6];
    if (sscanf(argv[4], "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) return usage();
    for (int d = 0; d < 6; d++) if (b[d] % 2 || (d >= 3 && b[d] <= 0)) { fprintf(stderr, "band: box must be even and positive\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75"))};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "band: cannot load axis\n"); return 1; }
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
    const int ffh = atoi(opt(argc, argv, "--fit-h", "8"));   /* field voxel edge, native voxels (even) */
    if (ff_dir && (ffh < 2 || ffh % 2 || b[3] % ffh || b[4] % ffh || b[5] % ffh)) { fprintf(stderr, "band: --fit-h must be even and divide the box\n"); return 2; }
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
   [--knot-dz 1024] [--quiet] [--out MODEL-DIR] [--winding ZARR-DIR [--winding-step 4] [--um 2.4]]
   The winding fitter (src/fit.h) on a directory of fit fields (ufsm band --fit-fields, or predict --fit-fields later). --codes: score
   against a winding_mod14 band raster (label grid, 2 native voxels per label voxel, code = 1 + round(18 q) mod 252 with q the mesh
   coordinate, q = U for dir +1): on every labelled voxel of the box U_fit - q must be one constant mod 14 turns. */
static int cmp_float(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static int arg_index(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }
static int cmd_fit(int argc, char **argv) {
    const char *dir = argv[2];
    char path[1400]; snprintf(path, sizeof path, "%s/fields.json", dir);
    size_t jn; FILE *jf = fopen(path, "rb"); if (!jf) { fprintf(stderr, "fit: no %s\n", path); return 1; }
    fseek(jf, 0, SEEK_END); jn = (size_t)ftell(jf); fseek(jf, 0, SEEK_SET); char *jt = malloc(jn + 1); if (fread(jt, 1, jn, jf) != jn) return 1; fclose(jf); jt[jn] = 0;
    json *j = json_parse(jt, jn); free(jt); if (!j) { fprintf(stderr, "fit: bad fields.json\n"); return 1; }
    long long org[3]; int shp[3];
    for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
    const double h = json_num(json_get(j, "h"), 8);
    const json *axj = json_get(j, "axis"); const int na = (int)axj->n;
    float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)json_num(json_at(json_at(axj, z), 0), 0); cx[z] = (float)json_num(json_at(json_at(axj, z), 1), 0); }
    json_free(j);
    const size_t n = (size_t)shp[0] * shp[1] * shp[2];
    static const char *nm[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    uint8_t *a[10];
    for (int k = 0; k < 10; k++) {
        snprintf(path, sizeof path, "%s/%s.raw", dir, nm[k]); a[k] = malloc(n);
        FILE *f = fopen(path, "rb"); if (!f || fread(a[k], 1, n, f) != n) { fprintf(stderr, "fit: cannot read %s\n", path); return 1; } fclose(f);
    }
    fit_fields ff = {shp[0], shp[1], shp[2], h, a[0], a[1], {a[2], a[3], a[4], a[5], a[6], a[7]}, a[8], a[9], cy, cx};
    fit_cfg c = fit_cfg_default();
    c.verbose = arg_index(argc, argv, "--quiet") < 0;
    c.dir = atoi(opt(argc, argv, "--dir", "0")); c.knot_dz = atoi(opt(argc, argv, "--knot-dz", "1024")); c.phase_stage = atoi(opt(argc, argv, "--phase-stage", "4"));
    if (*opt(argc, argv, "--lattice", "")) { c.nstage = 0; char *t = strdup(opt(argc, argv, "--lattice", "")); for (char *q = strtok(t, ","); q && c.nstage < 8; q = strtok(nullptr, ",")) c.lattice[c.nstage++] = atoi(q); free(t); }
    if (*opt(argc, argv, "--steps", "")) { int k = 0; char *t = strdup(opt(argc, argv, "--steps", "")); for (char *q = strtok(t, ","); q && k < 8; q = strtok(nullptr, ",")) c.steps[k++] = atoi(q); for (; k < c.nstage; k++) c.steps[k] = c.steps[k - 1]; free(t); }
    nn_init(atoi(opt(argc, argv, "--gpu", "0")));
    const double t0 = now();
    fit_model *m = fit_run(&ff, &c);
    if (!m) return 1;
    fprintf(stderr, "fit: dir %+d, %d knots, lattice %d (%dx%dx%d), loss %.5f in %.1fs\n", m->dir, m->nk, m->L, m->lz, m->ly, m->lx, m->loss[0], now() - t0);
    for (int d = 0; d < 3; d++) m->origin[d] = org[d];
    const char *mout = opt(argc, argv, "--out", nullptr);
    if (mout && fit_save(m, mout)) { fprintf(stderr, "fit: cannot save %s\n", mout); return 1; }
    const char *wout = opt(argc, argv, "--winding", nullptr);
    if (wout) {   /* the per-voxel winding number W * 256 (uint16, exact volcomp typed chunks) on a grid of --winding-step native voxels */
        const int ws = atoi(opt(argc, argv, "--winding-step", "4")); const double um = atof(opt(argc, argv, "--um", "2.4"));
        const int64_t gs[3] = {(int64_t)ceil(shp[0] * h / ws), (int64_t)ceil(shp[1] * h / ws), (int64_t)ceil(shp[2] * h / ws)};
        char lv[32], ld[1400], at[600]; z3w_level_name(um * ws, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", wout, lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding coordinate W * 256 (uint16; W = U - dir theta / 2 pi, sheets at integer W)\",\"dir\":%d,\"origin_zyx\":[%lld,%lld,%lld],\"step\":%d}}",
                 m->dir, org[0], org[1], org[2], ws);
        z3w *zw = z3w_create_typed(ld, gs, 128, 1, 0.f, 0.0, at);
        if (!zw || z3w_write_group(wout, (const double[]){um * ws}, 1, "ufsm-winding", at)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        uint16_t *sb = malloc((size_t)128 * 128 * 128 * 2);
        for (int64_t sz = 0; sz * 128 < gs[0]; sz++) for (int64_t sy = 0; sy * 128 < gs[1]; sy++) for (int64_t sx = 0; sx * 128 < gs[2]; sx++) {
            memset(sb, 0, (size_t)128 * 128 * 128 * 2);
            for (int a = 0; a < 128 && sz * 128 + a < gs[0]; a++) for (int b = 0; b < 128 && sy * 128 + b < gs[1]; b++) for (int c = 0; c < 128 && sx * 128 + c < gs[2]; c++) {
                const double W = fit_W(m, ((sz * 128 + a) + 0.5) * ws, ((sy * 128 + b) + 0.5) * ws, ((sx * 128 + c) + 0.5) * ws);
                const double v = W * 256; sb[((size_t)a * 128 + b) * 128 + c] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : lround(v));
            }
            if (z3w_write_shard(zw, sz, sy, sx, (const uint8_t *)sb, 8)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        }
        z3w_close(zw); free(sb);
        fprintf(stderr, "fit: winding %lldx%lldx%lld (step %d) written to %s\n", (long long)gs[0], (long long)gs[1], (long long)gs[2], ws, wout);
    }
    const int ci = arg_index(argc, argv, "--codes");
    if (ci > 0 && ci + 2 < argc) {   /* score against the band codes of the box */
        store *s = store_open(argv[ci + 1]); z3 *z = z3_open(s, argv[ci + 2], nullptr);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        int64_t lo[3], ln[3]; for (int d = 0; d < 3; d++) { lo[d] = org[d] / 2; ln[d] = (int64_t)(shp[d] * h / 2); }
        const size_t nl = (size_t)ln[0] * ln[1] * ln[2]; uint8_t *codes = malloc(nl);
        if (z3_read(z, lo, ln, codes, 8)) { fprintf(stderr, "fit: code read failed: %s\n", z3_error()); return 1; }
        enum { NB = 14 * 72 }; static size_t hist[NB]; memset(hist, 0, sizeof hist);
        float *dv = malloc(nl * sizeof(float)); size_t nd = 0;
        for (int64_t zz = 0; zz < ln[0]; zz++) for (int64_t yy = 0; yy < ln[1]; yy++) for (int64_t xx = 0; xx < ln[2]; xx++) {
            const uint8_t cd = codes[((size_t)zz * ln[1] + yy) * ln[2] + xx];
            if (cd < 1 || cd > 252) continue;
            const double pz = 2.0 * (zz + lo[0]) + 0.5 - org[0], py = 2.0 * (yy + lo[1]) + 0.5 - org[1], px = 2.0 * (xx + lo[2]) + 0.5 - org[2];
            if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
            const double cya = cy[(int)fmin(pz, na - 1)], cxa = cx[(int)fmin(pz, na - 1)];
            const double U = fit_W(m, pz, py, px) + m->dir * atan2(py - cya, px - cxa) / (2 * M_PI);
            double d = fmod(U - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14;
            dv[nd++] = (float)d; hist[(int)(d / 14 * NB) % NB]++;
        }
        size_t best = 0; int bi = 0;
        for (int k = 0; k < NB; k++) { size_t sacc = 0; for (int e = -18; e <= 18; e++) sacc += hist[(k + e + NB) % NB]; if (sacc > best) { best = sacc; bi = k; } }   /* +-0.25 turn window */
        const double mode = (bi + 0.5) * 14.0 / NB;
        size_t right = 0, close = 0; double mean = 0;
        for (size_t i = 0; i < nd; i++) { double e = dv[i] - mode; e -= 14 * round(e / 14); if (fabs(e) < 0.5) { right++; mean += e; } }
        mean = right ? mean / right : 0;   /* the gauge refined to the mean error of the right voxels */
        float *ae = malloc((nd + 1) * sizeof(float)); size_t na2 = 0;
        for (size_t i = 0; i < nd; i++) { double e = dv[i] - mode - mean; e -= 14 * round(e / 14); close += fabs(e) < 0.08; if (fabs(e) < 0.5) ae[na2++] = (float)fabs(e); }
        qsort(ae, na2, sizeof(float), cmp_float);
        printf("fit score: %zu labelled voxels, gauge %.3f, winding ids right %.4f, |error| < 0.08 turn %.4f, |error| median %.3f p90 %.3f turn\n", nd, mode + mean,
               nd ? (double)right / nd : 0, nd ? (double)close / nd : 0, na2 ? ae[na2 / 2] : 0, na2 ? ae[na2 * 9 / 10] : 0);
        free(ae);
        free(codes); free(dv); z3_close(z);
    }
    fit_free(m); for (int k = 0; k < 10; k++) free(a[k]); free(cy); free(cx);
    return 0;
}

/* ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4]
   One tifxyz per wrap of a fitted winding model (ufsm fit --out): wrap k is the level set W = k. Rows every step native voxels in
   z, columns every step / r_k radians of theta over the box's angular range (r_k: the wrap's radius at the box centre), each
   vertex the root of W = k along the ray from the umbilicus (src/fit.h fit_ray_root); vertices outside the box, without a root,
   or (--fields) in a field voxel without data are -1. Coordinates are global native voxels. */
static int cmd_wraps(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4]\n"); return 2; }
    fit_model *m = fit_load(argv[2]); if (!m) { fprintf(stderr, "wraps: cannot load %s\n", argv[2]); return 1; }
    const double step = atof(opt(argc, argv, "--step", "8")), um = atof(opt(argc, argv, "--um", "2.4"));
    const double Z = m->D * m->h, Y = m->H * m->h, X = m->W * m->h;
    uint8_t *wf = nullptr;   /* optional: the fields' data weight */
    const char *fd = opt(argc, argv, "--fields", nullptr);
    if (fd) {
        char p[1400]; snprintf(p, sizeof p, "%s/w.raw", fd); const size_t nf = (size_t)m->D * m->H * m->W; wf = malloc(nf);
        FILE *f = fopen(p, "rb"); if (!f || fread(wf, 1, nf, f) != nf) { fprintf(stderr, "wraps: cannot read %s\n", p); return 1; } fclose(f);
    }
    double wmin = 1e300, wmax = -1e300;   /* the wraps that meet the box */
    for (int a = 0; a <= 16; a++) for (int b = 0; b <= 16; b++) for (int c = 0; c <= 16; c++) {
        const double w = fit_W(m, Z * a / 16, Y * b / 16, X * c / 16); if (w < wmin) wmin = w; if (w > wmax) wmax = w;
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
        if (fit_ray_root(m, zc, thc, k, hypot(Y / 2 - cyc, X / 2 - cxc), &rk)) rk = hypot(Y / 2 - cyc, X / 2 - cxc);
        const double dth = step / fmax(rk, step);
        const int cols = (int)ceil((th1 - th0) / dth) + 1;
        const size_t nv = (size_t)rows * cols;
        float *px = malloc(nv * 4), *py = malloc(nv * 4), *pz = malloc(nv * 4); size_t valid = 0;
        for (int r = 0; r < rows; r++) {
            const double z = fmin(r * step, Z), cy = m->cy[(int)fmin(z, na - 1)], cx = m->cx[(int)fmin(z, na - 1)];
            double rprev = rk;
            for (int c = 0; c < cols; c++) {
                const size_t i = (size_t)r * cols + (cols - 1 - c);   /* column 0 = the largest theta side (upstream: outermost first) */
                const double th = th0 + c * dth; double rr;
                px[i] = py[i] = pz[i] = -1;
                if (fit_ray_root(m, z, th, k, rprev, &rr)) continue;
                const double y = cy + rr * sin(th), x = cx + rr * cos(th);
                if (wf) {
                    const int fz = (int)fmin(z / m->h, m->D - 1), fy = (int)fmin(y / m->h, m->H - 1), fx = (int)fmin(x / m->h, m->W - 1);
                    if (fy < 0 || fx < 0 || !wf[((size_t)fz * m->H + fy) * m->W + fx]) continue;
                }
                px[i] = (float)(x + m->origin[2]); py[i] = (float)(y + m->origin[1]); pz[i] = (float)(z + m->origin[0]); valid++; rprev = rr;
            }
        }
        if (valid > 16) {
            char wd[1400]; snprintf(wd, sizeof wd, "%s/w%03d", argv[3], k);
            if (tifxyz_write(wd, cols, rows, px, py, pz, step, um, k, "ufsm wraps (fitted winding level set)")) { fprintf(stderr, "wraps: cannot write %s\n", wd); return 1; }
            nwr++; nvalid_all += valid;
        }
        free(px); free(py); free(pz);
    }
    fprintf(stderr, "wraps: %d wraps (%d..%d) with %zu vertices written to %s\n", nwr, k0, k1, nvalid_all, argv[3]);
    fit_free(m); free(wf);
    return 0;
}

/* ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8]
   Per wrap (each <wraps-dir>/wNNN tifxyz): the vertex grid bilinearly interpolated to scale pixels per native voxel, the
   surface normal from the grid derivatives, the volume (an ink zarr level, uint8) sampled trilinearly at t = -depth..depth native
   voxels along it and max-composited into <out-dir>/wNNN.tif (uint8); --ct also the CT at t = 0 into wNNN_ct.tif. Pixels whose
   cell has an invalid corner are 0. The volume box under the wrap is read once (sub-block wraps). */
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
static int cmd_render(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8]\n"); return 2; }
    const int vi = arg_index(argc, argv, "--volume"), ci = arg_index(argc, argv, "--ct");
    if (vi < 0 || vi + 2 >= argc) { fprintf(stderr, "render: --volume ROOT KEY\n"); return 2; }
    const double scale = atof(opt(argc, argv, "--scale", "1")); const int depth = atoi(opt(argc, argv, "--depth", "8"));
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
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3) { free(X); free(Y); free(Z); continue; }
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        double step = 1; { FILE *f = fopen(p, "rb"); char b[4096] = {0}; if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); const char *q = strstr(b, "\"scale\": ["); if (q) step = 1 / atof(q + 10); } }
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (size_t i = 0; i < (size_t)w * h; i++) { if (Z[i] < 0) continue; const double v[3] = {Z[i], Y[i], X[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; } }
        if (lo[0] > hi[0]) { free(X); free(Y); free(Z); continue; }
        int64_t o[3], n[3];
        for (int d = 0; d < 3; d++) { o[d] = (int64_t)floor(lo[d]) - depth - 2; if (o[d] < 0) o[d] = 0; n[d] = (int64_t)ceil(hi[d]) + depth + 3 - o[d]; }
        uint8_t *vol = malloc((size_t)n[0] * n[1] * n[2]), *ct = cz ? malloc((size_t)n[0] * n[1] * n[2]) : nullptr;
        if (z3_read(vz, o, n, vol, 16) || (cz && z3_read(cz, o, n, ct, 16))) { fprintf(stderr, "render: read failed: %s\n", z3_error()); return 1; }
        const int ow = (int)((w - 1) * step * scale) + 1, oh = (int)((h - 1) * step * scale) + 1;
        uint8_t *img = calloc((size_t)ow * oh, 1), *cimg = cz ? calloc((size_t)ow * oh, 1) : nullptr;
        for (int py = 0; py < oh; py++) for (int px = 0; px < ow; px++) {
            const double gu = px / (step * scale), gv = py / (step * scale);
            int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
            const double fu = gu - c0, fv = gv - r0;
            const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
            if (Z[i00] < 0 || Z[i01] < 0 || Z[i10] < 0 || Z[i11] < 0) continue;
            const float *A[3] = {Z, Y, X}; double P[3], du[3], dv[3];
            for (int d = 0; d < 3; d++) {
                const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
                P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
                du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
            }
            double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
            const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
            for (int d = 0; d < 3; d++) N[d] /= nl;
            float m = 0;
            for (int t = -depth; t <= depth; t++) { const float v = trilin(vol, n, P[0] + t * N[0] - o[0], P[1] + t * N[1] - o[1], P[2] + t * N[2] - o[2]); if (v > m) m = v; }
            img[(size_t)py * ow + px] = (uint8_t)lroundf(m);
            if (cimg) cimg[(size_t)py * ow + px] = (uint8_t)lroundf(trilin(ct, n, P[0] - o[0], P[1] - o[1], P[2] - o[2]));
        }
        snprintf(p, sizeof p, "%s/%s.tif", argv[3], de->d_name);
        if (tiffw_write(p, ow, oh, 1, img, 1)) { fprintf(stderr, "render: cannot write %s\n", p); return 1; }
        if (cimg) { snprintf(p, sizeof p, "%s/%s_ct.tif", argv[3], de->d_name); if (tiffw_write(p, ow, oh, 1, cimg, 1)) return 1; }
        free(X); free(Y); free(Z); free(vol); free(ct); free(img); free(cimg); nw++;
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
