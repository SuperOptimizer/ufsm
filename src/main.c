/* ufsm CLI. */
#include "sample.h"
#include "sources.h"
#include "band.h"
#include "store.h"
#include "zarr3.h"
#include <dirent.h>
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
/* ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis umbilicus.json] [--radius 80] [--span 75] [--threads 8]
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
    uint8_t *codes = calloc(N, 1), *band = malloc(N);
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
    if (band_field(codes, ni, o, cy, cx, bp, band)) { fprintf(stderr, "band: out of memory\n"); return 1; }
    const int64_t on[3] = {b[3] / 2, b[4] / 2, b[5] / 2};
    FILE *f = fopen(argv[5], "wb");
    size_t unknown = 0;
    for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++) {
        const uint8_t *row = band + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo;
        for (int64_t xx = 0; xx < on[2]; xx++) unknown += row[xx] == BAND_UNKNOWN;
        if (!f || fwrite(row, 1, (size_t)on[2], f) != (size_t)on[2]) { fprintf(stderr, "band: write failed\n"); return 1; }
    }
    fclose(f);
    fprintf(stderr, "band field %lldx%lldx%lld (halo %d) in %.1fs, unknown %.1f%%\n", (long long)on[0], (long long)on[1], (long long)on[2], halo,
            now() - t0, 100.0 * unknown / ((double)on[0] * on[1] * on[2]));
    free(codes); free(band); free(cy); free(cx); z3_close(z);
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
        z3 *z = z3_open(s, key, cache);
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
        size_t nv = (size_t)n[0] * n[1] * n[2];
        uint8_t *buf = malloc(nv);
        double t0 = now();
        int rc = z3_read(z, o, n, buf, threads);
        double dt = now() - t0;
        if (rc) { fprintf(stderr, "read failed: %s\n", z3_error()); return 1; }
        size_t nz = 0; double sum = 0;
        for (size_t i = 0; i < nv; i++) { nz += buf[i] != 0; sum += buf[i]; }
        fprintf(stderr, "read %zu voxels in %.2fs (%.1f Mvox/s), nonzero %.1f%%, mean %.2f\n", nv, dt, nv / dt / 1e6,
                100.0 * nz / nv, sum / nv);
        if (!strcmp(cmd, "slice")) rc = write_pgm(out, buf, n[2], n[1]);
        else { FILE *f = fopen(out, "wb"); rc = f ? (fwrite(buf, 1, nv, f) == nv ? 0 : -1) : -1; if (f) fclose(f); }
        free(buf);
        z3_close(z);
        return rc ? 1 : 0;
    }
    return usage();
}
