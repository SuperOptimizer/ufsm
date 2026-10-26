/* ufsm ink3d-ext: dense inference of a published 3D ink network (src/ink3dext.h; gambiarra) over a box of a 2.4 um CT, written as
   prob254 zarr v3 levels on the scan's own pyramid grids, following paris4_3d_gambiarra/volume.py:

   - output grid (4.8 um) = level 1: voxel j covers level-0 [2j, 2j + 2); shape ceil(V / 2). Crop origins per axis =
     grid_axes(shape1, 64) (count ceil((n - 64) / 32) + 1, origin i = round((n - 64) / (count - 1) i), half to even), kept where the
     256^3 parent [2a - 64, 2a + 192) lies inside the level-0 array (the published run's global grid, so boxes agree with it);
   - per crop: p1 / p99 of the parent's 256-bin histogram (np.percentile linear), clip / scale the 128^3 crop [2a, 2a + 128), average
     pool 2 -> 64^3, the network -> 64^3 logits of output voxels [a, a + 64); crops whose 128^3 CT max is below --air are skipped;
   - blend: sum g l / sum g, g = Gaussian sigma 8 output voxels, peak 1; p = sigmoid;
   - 4.8 um output (--out-um 4.8): prob254 = round(254 p); 9.6 um output (level 2: voxel i covers level-0 [4i, 4i + 4)) = round(254
     mean of the known 4.8 p of its 2^3 block). 255 = unknown: outside the box, outside crop coverage, or only air / absent CT.

   The box (level-0 voxels) is widened to multiples of 4. Arrays have the full level shape (sparse: only shards touching the box are
   written, 9.6 um 512^3 shards, 4.8 um 1024^3 shards = one 9.6 shard each); a rerun into the same group merges (a known new value
   replaces the old one, unknown keeps it). Work: per 9.6 shard, y / x bands of --band 4.8 voxels, crop rows streamed in z through
   a ring of 128 output rows on the GPU; crops shared by two bands / shards are recomputed. */
#include "ink3dext.h"
#include "store.h"
#include "xnet.h"
#include "z3w.h"
#include "zarr2.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[1400]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "ink3d-ext: cannot write %s\n", path); return -1; }
    const size_t w = fwrite(p, 1, n, f); fclose(f); return w == n ? 0 : -1;
}

/* the 256^3 parent's histogram -> lut; returns the max of the 128^3 crop (centre) */
static void parent_lut(const uint8_t *base, size_t sy, size_t sx, float lut[256], double *lo, double *hi) {
    uint32_t h[4][256]; memset(h, 0, sizeof h);
    for (int z = 0; z < I3_PARENT; z++) for (int y = 0; y < I3_PARENT; y++) {
        const uint8_t *r = base + ((size_t)z * sy + y) * sx;
        for (int x = 0; x < I3_PARENT; x += 4) { h[0][r[x]]++; h[1][r[x + 1]]++; h[2][r[x + 2]]++; h[3][r[x + 3]]++; }
    }
    uint64_t hs[256]; for (int v = 0; v < 256; v++) hs[v] = (uint64_t)h[0][v] + h[1][v] + h[2][v] + h[3][v];
    i3_percentiles(hs, (uint64_t)I3_PARENT * I3_PARENT * I3_PARENT, lo, hi);
    i3_lut(*lo, *hi, lut);
}
static int crop_max(const uint8_t *base, size_t sy, size_t sx) {   /* base = the crop's first voxel */
    uint8_t m = 0;
    for (int z = 0; z < I3_CROP; z++) for (int y = 0; y < I3_CROP; y++) {
        const uint8_t *r = base + ((size_t)z * sy + y) * sx;
        for (int x = 0; x < I3_CROP; x++) m = r[x] > m ? r[x] : m;
        if (m == 255) return 255;
    }
    return m;
}
static void crop_copy(const uint8_t *base, size_t sy, size_t sx, uint8_t *dst) {
    for (int z = 0; z < I3_CROP; z++) for (int y = 0; y < I3_CROP; y++) memcpy(dst + ((size_t)z * I3_CROP + y) * I3_CROP, base + ((size_t)z * sy + y) * sx, I3_CROP);
}

/* --raw-in: N parents (u8 [N][256^3]) through the network (parity) */
static int raw_mode(gb_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr);
    const int N = atoi(opt(argc, argv, "--raw-n", "1")), reps = atoi(opt(argc, argv, "--reps", "1"));
    if (!in || !out || N < 1) { fprintf(stderr, "ink3d-ext --raw-in PARENTS.u8 --raw-n N --raw-out DIR\n"); return 2; }
    const size_t pv = (size_t)I3_PARENT * I3_PARENT * I3_PARENT, cv = (size_t)I3_CROP * I3_CROP * I3_CROP, ov = (size_t)I3_OUT * I3_OUT * I3_OUT;
    uint8_t *u = malloc(pv * N), *crops = malloc(cv * N); float *lut = malloc(sizeof(float) * 256 * N), *lg = malloc(ov * 4 * N);
    FILE *f = fopen(in, "rb"); if (!f || !u || fread(u, 1, pv * N, f) != pv * N) { fprintf(stderr, "ink3d-ext: %s is not %d x 256^3 bytes\n", in, N); return 1; } fclose(f);
    char st[64 * 1024]; size_t sl = 0;
    for (int i = 0; i < N; i++) {
        double lo, hi; const uint8_t *p = u + pv * i;
        parent_lut(p, I3_PARENT, I3_PARENT, lut + 256 * i, &lo, &hi);
        crop_copy(p + ((size_t)64 * I3_PARENT + 64) * I3_PARENT + 64, I3_PARENT, I3_PARENT, crops + cv * i);
        sl += snprintf(st + sl, sizeof st - sl, "%.9g %.9g\n", lo, hi);
    }
    uint8_t *cd = xmalloc_dev(cv * N); float *ld = xmalloc_dev(sizeof(float) * 256 * N);
    xh2d(cd, crops, cv * N); xh2d(ld, lut, sizeof(float) * 256 * N);
    const float *lgd = nullptr; double t0 = 0, t1 = 0;
    for (int r = 0; r < reps; r++) {
        xnet_sync(); t0 = now_s();
        if (gb_forward(net, cd, ld, N, &lgd)) return 1;
        xnet_sync(); t1 = now_s();
    }
    fprintf(stderr, "ink3d-ext: %d crops forward %.1f ms (%.1f ms / crop), %.1f GMAC / crop\n", N, (t1 - t0) * 1e3, (t1 - t0) * 1e3 / N, xnet_macs() / reps / N / 1e9);
    xd2h(lg, lgd, ov * 4 * N);
    { char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1; }
    if (write_file(out, "logits.f32", lg, ov * 4 * N) || write_file(out, "stats.txt", st, sl)) return 1;
    free(u); free(crops); free(lut); free(lg); xfree_dev(cd); xfree_dev(ld);
    return 0;
}

/* villa volume.grid_axes + the parent filter of crop_grid */
static int grid_axis(int64_t V0, int **out) {
    const int64_t n = (V0 + 1) / 2;
    int *a = malloc(sizeof(int) * (size_t)(n / 32 + 4)); int k = 0;
    if (n >= I3_OUT) {
        const int64_t count = (n - I3_OUT + 31) / 32 + 1;   /* ceil((n - 64) / 32) + 1 */
        for (int64_t i = 0; i < count; i++) {
            const int64_t s = count > 1 ? (int64_t)nearbyint((double)(n - I3_OUT) / (double)(count - 1) * (double)i) : 0;   /* Python round: half to even */
            if (s < 0 || s + I3_OUT > n) continue;
            if (2 * s - 64 < 0 || 2 * s + 192 > V0) continue;
            a[k++] = (int)s;
        }
    }
    *out = a; return k;
}

typedef struct {
    /* inputs */
    const uint8_t *slab; size_t sy, sx;                  /* CT slab [256][sy][sx], level-0 origin (zs, ys, xs) */
    int64_t zs, ys, xs;
    int oz; const int *oy, *ox; int nj;                  /* crops of this chunk: (oz, oy[j], ox[j]) */
    int air;
    /* outputs */
    uint8_t *crops; float *lut; int *keep;
    atomic_int next;
} prep_job;
static void *prep_worker(void *arg) {
    prep_job *J = arg;
    for (;;) {
        const int j = atomic_fetch_add(&J->next, 1); if (j >= J->nj) break;
        const int64_t pz = 2 * (int64_t)J->oz - 64 - J->zs, py = 2 * (int64_t)J->oy[j] - 64 - J->ys, px = 2 * (int64_t)J->ox[j] - 64 - J->xs;
        const uint8_t *pb = J->slab + ((size_t)pz * J->sy + py) * J->sx + px, *cb = pb + ((size_t)64 * J->sy + 64) * J->sx + 64;
        if (crop_max(cb, J->sy, J->sx) < J->air) { J->keep[j] = 0; continue; }
        double lo, hi; parent_lut(pb, J->sy, J->sx, J->lut + 256 * j, &lo, &hi);
        crop_copy(cb, J->sy, J->sx, J->crops + (size_t)I3_CROP * I3_CROP * I3_CROP * j);
        J->keep[j] = 1;
    }
    return nullptr;
}

typedef struct {
    z3 *ct; int nth, B, air, R;
    gb_net *net;
    int *A[3], nA[3];
    /* device */
    float *acc, *wsum; uint8_t *d48, *d96, *dcrop; float *dlut; int *dorg; unsigned long long *dhist;
    /* host */
    uint8_t *slab; size_t slab_cap; uint8_t *crops, *h48, *h96; float *lut; int *keep, *org;
    long long ncrop, nair; double tgpu_wait, tct, tprep;
} run_ctx;

/* emit rows [z0, z1) of a band into the shard buffers (b48: (2 SH)^3 at 4.8, origin s48; b96: SH^3, origin s96) */
static void emit(run_ctx *C, int TY, int TX, int z0, int z1, int y0, int x0, uint8_t *b48, const int64_t s48[3], uint8_t *b96, const int64_t s96[3], int SH) {
    if (z1 <= z0) return;
    i3_emit(C->acc, C->wsum, C->R, TY, TX, z0, z1, b48 ? C->d48 : nullptr, b96 ? C->d96 : nullptr, C->dhist);
    const int nz = z1 - z0, NY = (TY + 1) / 2, NX = (TX + 1) / 2, E = 2 * SH;
    if (b48) {
        xd2h(C->h48, C->d48, (size_t)nz * TY * TX);
        for (int z = 0; z < nz; z++) for (int y = 0; y < TY; y++)
            memcpy(b48 + (((size_t)(z0 + z - s48[0]) * E + (y0 + y - s48[1])) * E + (x0 - s48[2])), C->h48 + ((size_t)z * TY + y) * TX, TX);
    }
    if (b96) {
        const int nz2 = (nz + 1) / 2;
        xd2h(C->h96, C->d96, (size_t)nz2 * NY * NX);
        for (int z = 0; z < nz2; z++) for (int y = 0; y < NY; y++)
            memcpy(b96 + (((size_t)(z0 / 2 + z - s96[0]) * SH + (y0 / 2 + y - s96[1])) * SH + (x0 / 2 - s96[2])), C->h96 + ((size_t)z * NY + y) * NX, NX);
    }
}

/* one band: output (4.8) z [tz0, tz1), y [y0, y1), x [x0, x1) */
static int run_band(run_ctx *C, int tz0, int tz1, int y0, int y1, int x0, int x1, uint8_t *b48, const int64_t s48[3], uint8_t *b96, const int64_t s96[3], int SH) {
    int lo[3], hi[3]; const int r0[3] = {tz0, y0, x0}, r1[3] = {tz1, y1, x1};
    for (int a = 0; a < 3; a++) {
        lo[a] = C->nA[a]; hi[a] = -1;
        for (int i = 0; i < C->nA[a]; i++) if (C->A[a][i] < r1[a] && C->A[a][i] + I3_OUT > r0[a]) { if (i < lo[a]) lo[a] = i; hi[a] = i; }
        if (hi[a] < 0) return 0;   /* no crop: stays unknown */
    }
    const int TY = y1 - y0, TX = x1 - x0;
    const int64_t ys = 2 * (int64_t)C->A[1][lo[1]] - 64, xs = 2 * (int64_t)C->A[2][lo[2]] - 64;
    const size_t sy = (size_t)(2 * (int64_t)C->A[1][hi[1]] + 192 - ys), sx = (size_t)(2 * (int64_t)C->A[2][hi[2]] + 192 - xs);
    if ((size_t)I3_PARENT * sy * sx > C->slab_cap) { free(C->slab); C->slab_cap = (size_t)I3_PARENT * sy * sx; C->slab = malloc(C->slab_cap); if (!C->slab) { fprintf(stderr, "ink3d-ext: out of host memory (CT slab %.2f GB)\n", C->slab_cap / 1e9); return -1; } }
    const int ny = hi[1] - lo[1] + 1, nx = hi[2] - lo[2] + 1, nj = ny * nx;
    int *oy = malloc(sizeof(int) * nj), *ox = malloc(sizeof(int) * nj);
    for (int j = 0; j < nj; j++) { oy[j] = C->A[1][lo[1] + j / nx]; ox[j] = C->A[2][lo[2] + j % nx]; }
    int64_t zs = INT64_MIN; int fin = tz0;
    for (int iz = lo[0]; iz <= hi[0]; iz++) {
        const int oz = C->A[0][iz];
        /* rows below oz are final */
        int e1 = oz < tz1 ? oz : tz1; e1 &= ~1;
        if (e1 > fin) { emit(C, TY, TX, fin, e1, y0, x0, b48, s48, b96, s96, SH); fin = e1; }
        /* the CT slab [2 oz - 64, 2 oz + 192) */
        const double tc = now_s();
        const int64_t nzs = 2 * (int64_t)oz - 64; int64_t keep = 0;
        if (zs != INT64_MIN && nzs > zs && nzs - zs < I3_PARENT) {
            keep = I3_PARENT - (nzs - zs);
            memmove(C->slab, C->slab + (size_t)(nzs - zs) * sy * sx, (size_t)keep * sy * sx);
        }
        { const int64_t o[3] = {nzs + keep, ys, xs}, n[3] = {I3_PARENT - keep, (int64_t)sy, (int64_t)sx};
          if (z3_read(C->ct, o, n, C->slab + (size_t)keep * sy * sx, C->nth)) { fprintf(stderr, "ink3d-ext: CT read: %s\n", z3_error()); return -1; } }
        zs = nzs; C->tct += now_s() - tc;
        for (int j0 = 0; j0 < nj; j0 += C->B) {
            const int n = nj - j0 < C->B ? nj - j0 : C->B;
            const double tp = now_s();
            prep_job J = {.slab = C->slab, .sy = sy, .sx = sx, .zs = zs, .ys = ys, .xs = xs, .oz = oz, .oy = oy + j0, .ox = ox + j0, .nj = n, .air = C->air,
                          .crops = C->crops, .lut = C->lut, .keep = C->keep};
            atomic_init(&J.next, 0);
            const int nt = C->nth < n ? C->nth : n; pthread_t th[64];
            for (int t = 1; t < nt && t < 64; t++) pthread_create(&th[t], nullptr, prep_worker, &J);
            prep_worker(&J);
            for (int t = 1; t < nt && t < 64; t++) pthread_join(th[t], nullptr);
            int m = 0;   /* compact the kept crops */
            const size_t cv = (size_t)I3_CROP * I3_CROP * I3_CROP;
            for (int j = 0; j < n; j++) {
                if (!C->keep[j]) { C->nair++; continue; }
                if (m != j) { memcpy(C->crops + cv * m, C->crops + cv * j, cv); memcpy(C->lut + 256 * m, C->lut + 256 * j, 1024); }
                C->org[3 * m] = oz; C->org[3 * m + 1] = oy[j0 + j] - y0; C->org[3 * m + 2] = ox[j0 + j] - x0; m++;
            }
            C->tprep += now_s() - tp;
            if (!m) continue;
            const double tw = now_s();
            xh2d(C->dcrop, C->crops, cv * m);   /* waits for the previous chunk */
            C->tgpu_wait += now_s() - tw;
            xh2d(C->dlut, C->lut, (size_t)m * 1024); xh2d(C->dorg, C->org, (size_t)m * 12);
            const float *lg = nullptr;
            if (gb_forward(C->net, C->dcrop, C->dlut, m, &lg)) return -1;
            i3_blend(C->acc, C->wsum, C->R, TY, TX, lg, C->dorg, m, tz0, tz1);
            C->ncrop += m;
        }
    }
    emit(C, TY, TX, fin, tz1, y0, x0, b48, s48, b96, s96, SH);
    free(oy); free(ox);
    return xnet_check("ink3d-ext band");
}

static int level_exists(const char *group, const char *lv) {
    char p[1500]; struct stat st; snprintf(p, sizeof p, "%s/%s/zarr.json", group, lv); return !stat(p, &st);
}

int cmd_ink3d_ext(int argc, char **argv) {
    const char *model = opt(argc, argv, "--model", "gambiarra"), *wts = opt(argc, argv, "--weights", nullptr);
    if (!wts || strcmp(model, "gambiarra") || (!opt(argc, argv, "--raw-in", nullptr) && (!opt(argc, argv, "--box", nullptr) || !opt(argc, argv, "--out", nullptr) || !opt(argc, argv, "--ct", nullptr)))) {
        fprintf(stderr, "usage: ufsm ink3d-ext --model gambiarra --weights model.safetensors --ct ROOT KEY --box z,y,x,nz,ny,nx --out GROUP\n"
                        "         [--out-um 9.6[,4.8]] [--ct-um 2.4] [--batch 16] [--air 25] [--band 1024] [--threads 8] [--cache DIR] [--gpu 0] [--gpu-mem GB]\n"
                        "  published 3D ink network scrollprize/ink_paris4_3d_4.8um_gambiarra over a box of a 2.4 um CT (KEY = level 0; box in level-0\n"
                        "  voxels, widened to multiples of 4). Writes GROUP/<4 x ct-um> (and GROUP/<2 x ct-um> with 4.8): uint8 prob254 = round(254 p),\n"
                        "  255 unknown, on the scan's level-2 (level-1) grid, full level shape, 512^3 (1024^3) shards; reruns merge. Prints a logit histogram.\n"
                        "  ufsm ink3d-ext --weights W --raw-in PARENTS.u8 --raw-n N --raw-out DIR   N 256^3 parents -> logits.f32 [N][64^3] (parity)\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (opt(argc, argv, "--gpu-mem", nullptr)) xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "8")) * 1e9));
    gb_net *net = gb_load(wts); if (!net) return 1;
    if (opt(argc, argv, "--raw-in", nullptr)) { const int rc = raw_mode(net, argc, argv); gb_free(net); return rc; }

    const char *root = nullptr, *key = nullptr, *out = opt(argc, argv, "--out", nullptr);
    for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) { root = argv[i + 1]; key = argv[i + 2]; }
    if (!root) { fprintf(stderr, "ink3d-ext: --ct ROOT KEY\n"); return 2; }
    const double ctum = atof(opt(argc, argv, "--ct-um", "2.4"));
    const char *ous = opt(argc, argv, "--out-um", nullptr);
    int want48 = 0, want96 = 0;
    if (!ous) want96 = 1;
    else for (const char *p = ous; *p;) {
        char *e; const double v = strtod(p, &e); if (e == p) { fprintf(stderr, "ink3d-ext: --out-um %s\n", ous); return 2; }
        if (fabs(v - 2 * ctum) < 0.01 * ctum) want48 = 1; else if (fabs(v - 4 * ctum) < 0.01 * ctum) want96 = 1;
        else { fprintf(stderr, "ink3d-ext: --out-um %g is neither 2 x nor 4 x --ct-um %g\n", v, ctum); return 2; }
        p = *e == ',' ? e + 1 : e;
    }
    run_ctx C = {0};
    C.nth = atoi(opt(argc, argv, "--threads", "8")); C.B = atoi(opt(argc, argv, "--batch", "16")); C.air = atoi(opt(argc, argv, "--air", "25")); C.R = 128; C.net = net;
    const int SH = 512, band = atoi(opt(argc, argv, "--band", "1024"));
    if (C.nth < 1) C.nth = 1;
    if (C.B < 1 || band < 64 || band % 2 || band > 2 * SH) { fprintf(stderr, "ink3d-ext: --batch >= 1, --band even in 64 .. %d\n", 2 * SH); return 2; }
    store *st = store_open(root); C.ct = st ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!C.ct) { fprintf(stderr, "ink3d-ext: %s %s: %s\n", root, key, z3_error()); return 1; }
    const int64_t *V = z3_meta_of(C.ct)->shape;
    long long bv[6]; const char *box = opt(argc, argv, "--box", nullptr);
    if (sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld", bv, bv + 1, bv + 2, bv + 3, bv + 4, bv + 5) != 6) { fprintf(stderr, "ink3d-ext: --box z,y,x,nz,ny,nx\n"); return 2; }
    int64_t S1[3], S2[3], r48[2][3], r96[2][3];
    for (int a = 0; a < 3; a++) {
        if (bv[a] < 0 || bv[3 + a] < 1 || bv[a] + bv[3 + a] > V[a]) { fprintf(stderr, "ink3d-ext: the box leaves the volume (%lld %lld %lld)\n", (long long)V[0], (long long)V[1], (long long)V[2]); return 2; }
        S1[a] = (V[a] + 1) / 2; S2[a] = (V[a] + 3) / 4;
        const int64_t b0 = bv[a] / 4 * 4, b1 = (bv[a] + bv[3 + a] + 3) / 4 * 4;
        r48[0][a] = b0 / 2; r48[1][a] = (b1 / 2) < S1[a] ? b1 / 2 : S1[a];
        r96[0][a] = b0 / 4; r96[1][a] = (r48[1][a] + 1) / 2;
        C.nA[a] = grid_axis(V[a], &C.A[a]);
    }
    fprintf(stderr, "ink3d-ext: CT %lldx%lldx%lld, box (4.8) z %lld..%lld y %lld..%lld x %lld..%lld, crop grid %dx%dx%d, batch %d, band %d, air < %d\n",
            (long long)V[0], (long long)V[1], (long long)V[2], (long long)r48[0][0], (long long)r48[1][0], (long long)r48[0][1], (long long)r48[1][1],
            (long long)r48[0][2], (long long)r48[1][2], C.nA[0], C.nA[1], C.nA[2], C.B, band, C.air);

    /* outputs */
    char lv48[32], lv96[32], dir[1400], attrs[2048];
    z3w_level_name(2 * ctum, lv48, sizeof lv48); z3w_level_name(4 * ctum, lv96, sizeof lv96);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"ink3d-ext %s: sigmoid of the blended logits\",\"encoding\":\"prob254: p*254, 255 = unknown\","
             "\"weights\":\"%s\",\"source\":\"%s/%s\",\"box_zyx_level0\":[%lld,%lld,%lld,%lld,%lld,%lld]}}", model, wts, root, key, bv[0], bv[1], bv[2], bv[3], bv[4], bv[5]);
    { double ums[2]; int nl = 0;
      if (want48 || level_exists(out, lv48)) ums[nl++] = 2 * ctum;
      if (want96 || level_exists(out, lv96)) ums[nl++] = 4 * ctum;
      if (z3w_write_group(out, ums, nl, "ink3d-ext", attrs)) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; } }
    store *os = store_open(out);
    z3 *old48 = want48 && level_exists(out, lv48) ? z3_open(os, lv48, nullptr) : nullptr, *old96 = want96 && level_exists(out, lv96) ? z3_open(os, lv96, nullptr) : nullptr;
    z3w *w48 = nullptr, *w96 = nullptr;
    if (want48) { snprintf(dir, sizeof dir, "%s/%s", out, lv48); if (!(w48 = z3w_create(dir, S1, 2 * SH, 0, 255, attrs))) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; } }
    if (want96) { snprintf(dir, sizeof dir, "%s/%s", out, lv96); if (!(w96 = z3w_create(dir, S2, SH, 0, 255, attrs))) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; } }

    /* buffers */
    const size_t cv = (size_t)I3_CROP * I3_CROP * I3_CROP, ring = (size_t)C.R * band * band;
    C.acc = xmalloc_dev(ring * 4); C.wsum = xmalloc_dev(ring * 4); C.d48 = want48 ? xmalloc_dev(ring) : nullptr; C.d96 = xmalloc_dev(ring / 8 + band * band);
    C.dcrop = xmalloc_dev(cv * C.B); C.dlut = xmalloc_dev((size_t)C.B * 1024); C.dorg = xmalloc_dev((size_t)C.B * 12); C.dhist = xmalloc_dev(I3_HBINS * 8);
    C.crops = malloc(cv * C.B); C.lut = malloc((size_t)C.B * 1024); C.keep = malloc(sizeof(int) * C.B); C.org = malloc((size_t)C.B * 12);
    C.h48 = want48 ? malloc(ring) : nullptr; C.h96 = malloc(ring / 8 + band * band);
    uint8_t *b48 = want48 ? malloc((size_t)8 * SH * SH * SH) : nullptr, *b96 = want96 ? malloc((size_t)SH * SH * SH) : nullptr;
    if (!C.acc || !C.wsum || !C.d96 || !C.dcrop || !C.dlut || !C.dorg || !C.dhist || (want48 && (!C.d48 || !C.h48 || !b48)) || !C.crops || !C.lut || !C.h96 || (want96 && !b96)) {
        fprintf(stderr, "ink3d-ext: out of memory\n"); return 1;
    }
    {   /* zero the ring and the histogram */
        float *z = calloc(ring, 4); xh2d(C.acc, z, ring * 4); xh2d(C.wsum, z, ring * 4); free(z);
        unsigned long long hz[I3_HBINS] = {0}; xh2d(C.dhist, hz, sizeof hz);
    }

    const double t0 = now_s(); long long nshard = 0;
    for (int64_t sz = r96[0][0] / SH; sz * SH < r96[1][0]; sz++) for (int64_t sy = r96[0][1] / SH; sy * SH < r96[1][1]; sy++) for (int64_t sx = r96[0][2] / SH; sx * SH < r96[1][2]; sx++) {
        const int64_t s96[3] = {sz * SH, sy * SH, sx * SH}, s48[3] = {2 * sz * SH, 2 * sy * SH, 2 * sx * SH}, so96[3] = {sz, sy, sx}, sn96[3] = {SH, SH, SH}, sn48[3] = {2 * SH, 2 * SH, 2 * SH};
        int64_t t48[2][3];
        for (int a = 0; a < 3; a++) { t48[0][a] = s48[a] > r48[0][a] ? s48[a] : r48[0][a]; t48[1][a] = s48[a] + 2 * SH < r48[1][a] ? s48[a] + 2 * SH : r48[1][a]; }
        if (b96) { if (old96 && z3_shard_present(old96, sz, sy, sx) == 1) { if (z3_read(old96, s96, sn96, b96, C.nth)) { fprintf(stderr, "ink3d-ext: %s\n", z3_error()); return 1; } } else memset(b96, 255, (size_t)SH * SH * SH); }
        if (b48) { if (old48 && z3_shard_present(old48, sz, sy, sx) == 1) { if (z3_read(old48, s48, sn48, b48, C.nth)) { fprintf(stderr, "ink3d-ext: %s\n", z3_error()); return 1; } } else memset(b48, 255, (size_t)8 * SH * SH * SH); }
        /* merge: the bands write 255 where they know nothing; those voxels take the old shard's value afterwards */
        uint8_t *o96 = nullptr, *o48 = nullptr;
        if (old96 && b96) { o96 = malloc((size_t)SH * SH * SH); memcpy(o96, b96, (size_t)SH * SH * SH); }
        if (old48 && b48) { o48 = malloc((size_t)8 * SH * SH * SH); memcpy(o48, b48, (size_t)8 * SH * SH * SH); }
        for (int64_t y0 = t48[0][1]; y0 < t48[1][1]; y0 += band) for (int64_t x0 = t48[0][2]; x0 < t48[1][2]; x0 += band) {
            const int64_t y1 = y0 + band < t48[1][1] ? y0 + band : t48[1][1], x1 = x0 + band < t48[1][2] ? x0 + band : t48[1][2];
            if (run_band(&C, (int)t48[0][0], (int)t48[1][0], (int)y0, (int)y1, (int)x0, (int)x1, b48, s48, b96, s96, SH)) return 1;
            fprintf(stderr, "ink3d-ext: shard %lld,%lld,%lld band y %lld x %lld: %lld crops (+%lld air), %.1f s\n", (long long)sz, (long long)sy, (long long)sx,
                    (long long)y0, (long long)x0, C.ncrop, C.nair, now_s() - t0);
        }
        if (o96) { for (size_t i = 0; i < (size_t)SH * SH * SH; i++) if (b96[i] == 255) b96[i] = o96[i]; free(o96); }
        if (o48) { for (size_t i = 0; i < (size_t)8 * SH * SH * SH; i++) if (b48[i] == 255) b48[i] = o48[i]; free(o48); }
        if (b96 && z3w_write_shard(w96, so96[0], so96[1], so96[2], b96, C.nth)) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; }
        if (b48 && z3w_write_shard(w48, sz, sy, sx, b48, C.nth)) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; }
        nshard++;
    }
    if ((w96 && z3w_close(w96)) || (w48 && z3w_close(w48))) { fprintf(stderr, "ink3d-ext: %s\n", z3w_error()); return 1; }
    const double el = now_s() - t0;
    unsigned long long hist[I3_HBINS]; xd2h(hist, C.dhist, sizeof hist);
    unsigned long long tot = 0; for (int k = 0; k < I3_HBINS; k++) tot += hist[k];
    const double nv48 = (double)(r48[1][0] - r48[0][0]) * (r48[1][1] - r48[0][1]) * (r48[1][2] - r48[0][2]);
    fprintf(stderr, "ink3d-ext: logit histogram of the %llu known 4.8 um voxels (bin: count, %%):\n", tot);
    for (int k = 0; k < I3_HBINS; k++) {
        if (!hist[k]) continue;
        if (k == 0) fprintf(stderr, "  < -16      %12llu %6.2f\n", hist[k], 100.0 * hist[k] / tot);
        else if (k == I3_HBINS - 1) fprintf(stderr, "  >= 16      %12llu %6.2f\n", hist[k], 100.0 * hist[k] / tot);
        else fprintf(stderr, "  [%5.1f,%5.1f) %12llu %6.2f\n", -16 + 0.5 * (k - 1), -16 + 0.5 * k, hist[k], 100.0 * hist[k] / tot);
    }
    fprintf(stderr, "ink3d-ext: done: %lld shards, %lld crops (+%lld air) in %.1f s (CT read %.1f s, prep %.1f s, GPU wait %.1f s); %.0f ms / crop; "
                    "%.2f Mvox/s at 4.8 um (box %.0f Mvox), %.2f known\n",
            nshard, C.ncrop, C.nair, el, C.tct, C.tprep, C.tgpu_wait, C.ncrop ? 1e3 * el / C.ncrop : 0.0, nv48 / el / 1e6, nv48 / 1e6, nv48 > 0 ? tot / nv48 : 0.0);
    xfree_dev(C.acc); xfree_dev(C.wsum); xfree_dev(C.d48); xfree_dev(C.d96); xfree_dev(C.dcrop); xfree_dev(C.dlut); xfree_dev(C.dorg); xfree_dev(C.dhist);
    free(C.crops); free(C.lut); free(C.keep); free(C.org); free(C.h48); free(C.h96); free(C.slab); free(b48); free(b96);
    for (int a = 0; a < 3; a++) free(C.A[a]);
    if (old48) z3_close(old48); if (old96) z3_close(old96);
    z3_close(C.ct); gb_free(net);
    return 0;
}

/* ufsm vol-cmp: two uint8 volumes (zarr v3 or v2, local or https) over one box: the joint 256 x 256 histogram of the voxels known in
   both (A != --a-unknown, B != --b-unknown and B > --b-floor), Pearson and Spearman (ranks from the marginals, ties averaged)
   correlations, and the value mapping both ways (count, mean / p10 / p50 / p90 of the other volume per value bin). */
typedef struct { z3 *a3; z2 *a2; } vsrc;
static int vsrc_open(vsrc *v, const char *root, const char *key, const char *cache) {
    store *s = store_open(root); if (!s) return -1;
    v->a3 = z3_open(s, key, cache); if (v->a3) return 0;
    v->a2 = z2_open(s, key, cache); return v->a2 ? 0 : -1;
}
static int vsrc_read(vsrc *v, const int64_t o[3], const int64_t n[3], uint8_t *out, int nth) {
    return v->a3 ? z3_read(v->a3, o, n, out, nth) : z2_read(v->a2, o, n, out, nth);
}
static double hq(const double *h, double tot, double q) {   /* quantile of a 256-bin histogram */
    double c = 0; for (int v = 0; v < 256; v++) { c += h[v]; if (c >= q * tot) return v; } return 255;
}
int cmd_vol_cmp(int argc, char **argv) {
    if (argc < 7 || !opt(argc, argv, "--box", nullptr)) {
        fprintf(stderr, "usage: ufsm vol-cmp A_ROOT A_KEY B_ROOT B_KEY --box z,y,x,nz,ny,nx [--a-unknown 255] [--b-unknown -1] [--b-floor -1] [--bins 16]\n"
                        "         [--threads 8] [--cache DIR]   uint8 volumes (zarr v3 / v2): correlations + value mapping over the jointly known voxels\n");
        return 2;
    }
    const int au = atoi(opt(argc, argv, "--a-unknown", "255")), bu = atoi(opt(argc, argv, "--b-unknown", "-1")), bf = atoi(opt(argc, argv, "--b-floor", "-1"));
    const int nb = atoi(opt(argc, argv, "--bins", "16")), nth = atoi(opt(argc, argv, "--threads", "8")); const char *cache = opt(argc, argv, "--cache", nullptr);
    vsrc A = {0}, Bv = {0};
    if (vsrc_open(&A, argv[2], argv[3], cache)) { fprintf(stderr, "vol-cmp: %s %s: %s %s\n", argv[2], argv[3], z3_error(), z2_error()); return 1; }
    if (vsrc_open(&Bv, argv[4], argv[5], cache)) { fprintf(stderr, "vol-cmp: %s %s: %s %s\n", argv[4], argv[5], z3_error(), z2_error()); return 1; }
    long long b[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "vol-cmp: --box\n"); return 2; }
    double *J = calloc(256 * 256, sizeof(double)); long long nall = 0;
    const int64_t SL = 64;   /* z slabs */
    for (int64_t z = 0; z < b[3]; z += SL) {
        const int64_t o[3] = {b[0] + z, b[1], b[2]}, n[3] = {z + SL < b[3] ? SL : b[3] - z, b[4], b[5]};
        const size_t nv = (size_t)n[0] * n[1] * n[2]; uint8_t *ua = malloc(nv), *ub = malloc(nv);
        if (!ua || !ub || vsrc_read(&A, o, n, ua, nth) || vsrc_read(&Bv, o, n, ub, nth)) { fprintf(stderr, "vol-cmp: read failed: %s %s\n", z3_error(), z2_error()); return 1; }
        for (size_t i = 0; i < nv; i++) { nall++; if (ua[i] == au || ub[i] == bu || ub[i] <= bf) continue; J[ua[i] * 256 + ub[i]] += 1; }
        free(ua); free(ub);
    }
    double ha[256] = {0}, hb[256] = {0}, tot = 0;
    for (int i = 0; i < 256; i++) for (int k = 0; k < 256; k++) { ha[i] += J[i * 256 + k]; hb[k] += J[i * 256 + k]; tot += J[i * 256 + k]; }
    if (tot < 2) { fprintf(stderr, "vol-cmp: %lld voxels, none known in both\n", nall); return 1; }
    double ra[256], rb[256], c = 0;   /* mid ranks */
    for (int v = 0; v < 256; v++) { ra[v] = c + (ha[v] + 1) / 2; c += ha[v]; }
    c = 0; for (int v = 0; v < 256; v++) { rb[v] = c + (hb[v] + 1) / 2; c += hb[v]; }
    double ma = 0, mb = 0, mra = 0, mrb = 0;
    for (int v = 0; v < 256; v++) { ma += v * ha[v]; mb += v * hb[v]; mra += ra[v] * ha[v]; mrb += rb[v] * hb[v]; }
    ma /= tot; mb /= tot; mra /= tot; mrb /= tot;
    double sab = 0, saa = 0, sbb = 0, rab = 0, raa = 0, rbb = 0;
    for (int i = 0; i < 256; i++) for (int k = 0; k < 256; k++) {
        const double w = J[i * 256 + k]; if (!w) continue;
        sab += w * (i - ma) * (k - mb); saa += w * (i - ma) * (i - ma); sbb += w * (k - mb) * (k - mb);
        rab += w * (ra[i] - mra) * (rb[k] - mrb); raa += w * (ra[i] - mra) * (ra[i] - mra); rbb += w * (rb[k] - mrb) * (rb[k] - mrb);
    }
    printf("vol-cmp: %lld voxels, %.0f known in both (%.1f%%); A mean %.2f, B mean %.2f; Pearson %.4f, Spearman %.4f\n", nall, tot, 100 * tot / nall, ma, mb,
           sab / sqrt(saa * sbb), rab / sqrt(raa * rbb));
    for (int side = 0; side < 2; side++) {
        printf("%s bin        count      %s mean   p10   p50   p90\n", side ? "A" : "B", side ? "B" : "A");
        for (int g = 0; g < nb; g++) {
            const int v0 = g * 256 / nb, v1 = (g + 1) * 256 / nb; double h[256] = {0}, n = 0, s = 0;
            for (int u = v0; u < v1; u++) for (int w = 0; w < 256; w++) { const double x = side ? J[u * 256 + w] : J[w * 256 + u]; h[w] += x; n += x; s += x * w; }
            if (!n) continue;
            printf("  %3d..%3d %12.0f   %7.2f %5.0f %5.0f %5.0f\n", v0, v1 - 1, n, s / n, hq(h, n, 0.1), hq(h, n, 0.5), hq(h, n, 0.9));
        }
    }
    free(J);
    return 0;
}
