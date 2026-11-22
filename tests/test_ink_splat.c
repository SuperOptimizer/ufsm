/* ufsm ink-splat round trip: a synthetic curved surface (tifxyz), a 2D ink label + supervision mask on its canvas and a CT with a
   papyrus slab around the surface and a void hole. Splat, then sample the written zarr back along each pixel's normal and recover
   the label: ink 254 / no ink 0 inside the band on papyrus, unknown (255) outside the band, in the hole, and at unsupervised pixels;
   --flip moves a one-sided band to the other side; the coarse level keeps the band (known-max); --merge combines by max. */
#include "inksplat.h"
#include "inkflygpu.h"
#include "dcache.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include "fm_tol.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <time.h>

enum { NV = 256, GW = 40, GH = 44 };
static const double STEP = 4.0;   /* voxels per grid step (scale 0.25) */
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static void surf_at(double u, double v, double P[3]) {   /* grid (u, v) -> (z, y, x) */
    const double x = 30 + STEP * u, y = 28 + STEP * v;
    P[0] = 110 + 0.15 * x + 6 * sin(y / 25.0); P[1] = y; P[2] = x;
}
static void normal_at(double u, double v, double N[3]) {   /* du x dv as ink-splat ((z, y, x) components) */
    const double e = 1e-3; double a[3], b[3], c[3], d[3], du[3], dv[3];
    surf_at(u - e, v, a); surf_at(u + e, v, b); surf_at(u, v - e, c); surf_at(u, v + e, d);
    for (int k = 0; k < 3; k++) { du[k] = (b[k] - a[k]) / (2 * e); dv[k] = (d[k] - c[k]) / (2 * e); }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); for (int k = 0; k < 3; k++) N[k] /= l;
}
static int ink_at(int r, int c) { const double dr = r - 80, dc = c - 70; return dr * dr + dc * dc < 30 * 30 || (c >= 110 && c < 122); }
static int sup_at(int r, int c) { return !(r >= 120 && r < 140); }
int cmd_ink_score(int argc, char **argv);
/* ufsm ink-score in a child (it prints its --row on stdout): the row's F1 best and AUC (ops audit #17: the scoring commands had no test) */
static int ink_score(const char *dir, const char *map, const char *mask, double *f1, double *auc) {
    char out[512], sp[512], lp[512], mp[512], kp[512]; snprintf(out, sizeof out, "%s/score.tsv", dir); snprintf(sp, sizeof sp, "%s/seg.tifxyz", dir);
    snprintf(lp, sizeof lp, "%s/inklabels.tif", dir); snprintf(mp, sizeof mp, "m=%s", map); snprintf(kp, sizeof kp, "%s", mask ? mask : "");
    fflush(stdout); fflush(stderr);
    const pid_t c = fork();
    if (c == 0) {
        const int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) _exit(9); dup2(fd, 1);
        char *av[] = {"ufsm", "ink-score", "--surface", sp, "--labels", lp, "--map", mp, "--row", "t", "--threads", "2", mask ? "--mask" : nullptr, kp, nullptr};
        _exit(cmd_ink_score(mask ? 14 : 12, av));
    }
    int st = 0; waitpid(c, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) return -1;
    FILE *f = fopen(out, "r"); if (!f) return -1;
    char nm[64], tag[64]; unsigned long long n; double ink, thr, f05, p05, r05;
    const int k = fscanf(f, "%63s %63s %llu %lf %lf %lf %lf %lf %lf %lf", tag, nm, &n, &ink, f1, &thr, &f05, &p05, &r05, auc); fclose(f);
    return k == 10 ? 0 : -1;
}
static int hole_at(int64_t z, int64_t y, int64_t x) { return y >= 40 && y < 60 && x >= 120 && x < 150; }

int main(void) {
    char dir[] = "/tmp/ufsm_inksplat_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    char p[512], q[512];
    /* surface */
    float *X = malloc(GW * GH * 4), *Y = malloc(GW * GH * 4), *Z = malloc(GW * GH * 4);
    for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) { double P[3]; surf_at(u, v, P); X[v * GW + u] = (float)P[2]; Y[v * GW + u] = (float)P[1]; Z[v * GW + u] = (float)P[0]; }
    snprintf(p, sizeof p, "%s/seg.tifxyz", dir);
    if (tifxyz_write(p, GW, GH, X, Y, Z, STEP, 1.0, -1, "test")) { fprintf(stderr, "tifxyz_write failed\n"); return 1; }
    /* labels on the canvas (GW / scale x GH / scale pixels) */
    const int CW = (int)(GW * STEP), CHh = (int)(GH * STEP);
    uint8_t *lab = malloc((size_t)CW * CHh), *msk = malloc((size_t)CW * CHh);
    for (int r = 0; r < CHh; r++) for (int c = 0; c < CW; c++) { lab[r * CW + c] = ink_at(r, c) ? 255 : 0; msk[r * CW + c] = sup_at(r, c) ? 255 : 0; }
    snprintf(p, sizeof p, "%s/inklabels.tif", dir); tiffw_write(p, CW, CHh, 1, lab, 1);
    snprintf(p, sizeof p, "%s/mask.tif", dir); tiffw_write(p, CW, CHh, 1, msk, 1);
    /* CT: papyrus (200) where the distance along the local normal is in [-3, 5] (approximately: by the vertical offset scaled by
       the normal's z component), void (20) elsewhere and in the hole */
    uint8_t *ct = malloc((size_t)NV * NV * NV);
    for (int64_t z = 0; z < NV; z++) for (int64_t y = 0; y < NV; y++) for (int64_t x = 0; x < NV; x++) {
        const double u = (x - 30) / STEP, v = (y - 28) / STEP; double P[3], N[3]; surf_at(u, v, P); normal_at(u, v, N);
        const double d = (z - P[0]) * N[0];   /* signed distance along N (N[0] dominates) */
        ct[(z * NV + y) * NV + x] = hole_at(z, y, x) ? 20 : (d >= -6 && d <= 6) ? 200 : 20;
    }
    snprintf(p, sizeof p, "%s/ct.zarr", dir);
    { snprintf(q, sizeof q, "%s/1", p); const int64_t sh[3] = {NV, NV, NV}; z3w *w = z3w_create(q, sh, 256, 0, 0, nullptr);
      if (!w || z3w_write_shard(w, 0, 0, 0, ct, 4) || z3w_close(w)) { fprintf(stderr, "CT write failed\n"); return 1; }
      const double um[1] = {1.0}; z3w_write_group(p, um, 1, "ct", nullptr); }
    store_global_init();

    /* splat, band [-2, 2] */
    inksplat_cfg c; inksplat_defaults(&c);
    char out[512], surf[512], lp[512], mp[512];
    snprintf(out, sizeof out, "%s/out.zarr", dir); snprintf(surf, sizeof surf, "%s/seg.tifxyz", dir);
    snprintf(lp, sizeof lp, "%s/inklabels.tif", dir); snprintf(mp, sizeof mp, "%s/mask.tif", dir);
    c.out = out; c.surface = surf; c.labels = lp; c.mask = mp; c.ct_root = dir; c.ct_key = "ct.zarr"; c.ct_um = 1; c.level = 0; c.iso = 100;
    c.band_a = -2; c.band_b = 2; c.threads = 4; c.coarse = 2;
    CHECK(inksplat_run(&c) == 0, "splat");
    store *os = store_open(out); z3 *o = z3_open(os, "1", nullptr), *o2 = z3_open(os, "2", nullptr);
    CHECK(o && o2, "open output levels: %s", z3_error());
    if (!o || !o2) return 1;
    uint8_t *ob = malloc((size_t)NV * NV * NV); const int64_t z0[3] = {0, 0, 0}, nn[3] = {NV, NV, NV};
    CHECK(!z3_read(o, z0, nn, ob, 4), "read output");
    long n_in = 0, ok_in = 0, n_out = 0, ok_out = 0, n_hole = 0, ok_hole = 0, n_uns = 0, ok_uns = 0;
    for (int r = 2; r < CHh - 6; r++) for (int cc = 2; cc < CW - 6; cc++) {
        const int ink = ink_at(r, cc), edge = ink != ink_at(r - 2, cc) || ink != ink_at(r + 2, cc) || ink != ink_at(r, cc - 2) || ink != ink_at(r, cc + 2);
        const int sup = sup_at(r, cc), sedge = sup != sup_at(r - 3, cc) || sup != sup_at(r + 3, cc);
        if (edge || sedge) continue;
        double P[3], N[3]; surf_at(cc / STEP, r / STEP, P); normal_at(cc / STEP, r / STEP, N);
        for (int t = -5; t <= 5; t++) {
            int64_t vi[3]; for (int d = 0; d < 3; d++) vi[d] = (int64_t)floor(P[d] + t * N[d] + 0.5);
            if (vi[0] < 0 || vi[0] >= NV || vi[1] < 0 || vi[1] >= NV || vi[2] < 0 || vi[2] >= NV) continue;
            const int hole = hole_at(vi[0], vi[1], vi[2]) || hole_at(vi[0], vi[1] - 1, vi[2] - 1) || hole_at(vi[0], vi[1] + 1, vi[2] + 1);
            const uint8_t v = ob[(vi[0] * NV + vi[1]) * NV + vi[2]];
            if (!sup) { if (abs(t) <= 1) { n_uns++; ok_uns += v == 255; } continue; }
            if (hole) { if (hole_at(vi[0], vi[1], vi[2]) && abs(t) <= 1) { n_hole++; ok_hole += v == 255; } continue; }
            if (abs(t) <= 1) { n_in++; ok_in += v == (ink ? 254 : 0); }
            else if (abs(t) >= 4) { n_out++; ok_out += v == 255; }
        }
    }
    fprintf(stderr, "in band %ld/%ld, outside %ld/%ld, hole %ld/%ld, unsupervised %ld/%ld\n", ok_in, n_in, ok_out, n_out, ok_hole, n_hole, ok_uns, n_uns);
    CHECK(n_in > 20000 && ok_in >= 0.995 * n_in, "band recovery %ld/%ld", ok_in, n_in);
    CHECK(n_out > 10000 && ok_out >= 0.995 * n_out, "outside band unknown %ld/%ld", ok_out, n_out);
    CHECK(n_hole > 50 && ok_hole == n_hole, "hole unknown %ld/%ld", ok_hole, n_hole);
    CHECK(n_uns > 1000 && ok_uns >= 0.995 * n_uns, "unsupervised unknown %ld/%ld", ok_uns, n_uns);
    /* coarse level: known-max keeps ink */
    { const int64_t h[3] = {NV / 2, NV / 2, NV / 2}; uint8_t *cb = malloc((size_t)h[0] * h[1] * h[2]); CHECK(!z3_read(o2, z0, h, cb, 4), "read level 2");
      long ni = 0; for (size_t k = 0; k < (size_t)h[0] * h[1] * h[2]; k++) ni += cb[k] == 254;
      long nf = 0; for (size_t k = 0; k < (size_t)NV * NV * NV; k++) nf += ob[k] == 254;
      CHECK(ni > nf / 16, "coarse ink voxels %ld (fine %ld)", ni, nf); free(cb); }
    z3_close(o); z3_close(o2);

    /* one-sided band [0, 3]: labelled on +N only; flipped: on -N only (fresh outputs) */
    for (int flip = 0; flip < 2; flip++) {
        char o3[512]; snprintf(o3, sizeof o3, "%s/side%d.zarr", dir, flip);
        inksplat_cfg c2 = c; c2.out = o3; c2.band_a = 0; c2.band_b = 3; c2.flip = flip; c2.coarse = 0;
        CHECK(inksplat_run(&c2) == 0, "splat side %d", flip);
        store *s3 = store_open(o3); z3 *z = z3_open(s3, "1", nullptr); if (!z) { fails++; continue; }
        CHECK(!z3_read(z, z0, nn, ob, 4), "read side");
        long pos = 0, neg = 0, np = 0, ng = 0;
        for (int r = 4; r < 110; r += 3) for (int cc = 4; cc < 100; cc += 3) {
            double P[3], N[3]; surf_at(cc / STEP, r / STEP, P); normal_at(cc / STEP, r / STEP, N);
            int64_t a[3], b[3]; for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor(P[d] + 2 * N[d] + 0.5); b[d] = (int64_t)floor(P[d] - 2 * N[d] + 0.5); }
            if (hole_at(a[0], a[1], a[2]) || hole_at(b[0], b[1], b[2])) continue;
            np++; pos += ob[(a[0] * NV + a[1]) * NV + a[2]] != 255; ng++; neg += ob[(b[0] * NV + b[1]) * NV + b[2]] != 255;
        }
        fprintf(stderr, "flip %d: +N labelled %ld/%ld, -N labelled %ld/%ld\n", flip, pos, np, neg, ng);
        CHECK(flip ? (neg >= 0.99 * ng && pos <= 0.01 * np) : (pos >= 0.99 * np && neg <= 0.01 * ng), "side %d", flip);
        z3_close(z); store_close(s3);
    }

    /* --face -3,0: the band follows the papyrus slab's open face (d = +6 on +N; flipped: d = -6 on -N), not the mesh: known at
       |d| 4..5 on the face side, unknown at the mesh (|d| <= 1) and on the other side */
    for (int flip = 0; flip < 2; flip++) {
        char o6[512]; snprintf(o6, sizeof o6, "%s/face%d.zarr", dir, flip);
        inksplat_cfg c6 = c; c6.out = o6; c6.coarse = 0; c6.flip = flip; c6.face = 1; c6.face_a = -3; c6.face_b = 0;
        CHECK(inksplat_run(&c6) == 0, "face splat %d", flip);
        store *s6 = store_open(o6); z3 *z = z3_open(s6, "1", nullptr);
        CHECK(z && !z3_read(z, z0, nn, ob, 4), "read face");
        long nf = 0, okf = 0, nm = 0, okm = 0, no = 0, oko = 0; const double sg = flip ? -1 : 1;
        for (int r = 4; r < 110; r += 2) for (int cc = 4; cc < 100; cc += 2) {
            if (ink_at(r, cc) != ink_at(r - 3, cc) || ink_at(r, cc) != ink_at(r + 3, cc) || ink_at(r, cc) != ink_at(r, cc - 3) || ink_at(r, cc) != ink_at(r, cc + 3)) continue;
            double P[3], N[3]; surf_at(cc / STEP, r / STEP, P); normal_at(cc / STEP, r / STEP, N);
            for (int t = -5; t <= 5; t++) {
                int64_t vi[3]; for (int d = 0; d < 3; d++) vi[d] = (int64_t)floor(P[d] + sg * t * N[d] + 0.5);
                if (hole_at(vi[0], vi[1], vi[2]) || hole_at(vi[0], vi[1] - 1, vi[2] - 1) || hole_at(vi[0], vi[1] + 1, vi[2] + 1)) continue;
                const uint8_t v = ob[(vi[0] * NV + vi[1]) * NV + vi[2]];
                if (t >= 4) { nf++; okf += v == (ink_at(r, cc) ? 254 : 0); } else if (abs(t) <= 1) { nm++; okm += v == 255; } else if (t <= -3) { no++; oko += v == 255; }
            }
        }
        fprintf(stderr, "face flip %d: face band %ld/%ld, mesh unknown %ld/%ld, far side unknown %ld/%ld\n", flip, okf, nf, okm, nm, oko, no);
        CHECK(nf > 1000 && okf >= 0.98 * nf, "face band %d: %ld/%ld", flip, okf, nf);
        CHECK(nm > 1000 && okm >= 0.99 * nm, "face: mesh unknown %d: %ld/%ld", flip, okm, nm);
        CHECK(no > 1000 && oko >= 0.99 * no, "face: far side unknown %d: %ld/%ld", flip, oko, no);
        if (z) z3_close(z);
        store_close(s6);
    }
    /* on the fly: the derived array over the same window (two crops, merged) == the stored --face splat of those two crops */
    {
        char sj[512], js[2048], o7[512]; snprintf(sj, sizeof sj, "%s/segs.json", dir); snprintf(o7, sizeof o7, "%s/face2.zarr", dir);
        snprintf(js, sizeof js, "{\"segments\": [{\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\", \"crop\": [0, 0, %d, %d]},"
                 " {\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\", \"crop\": [0, %d, %d, %d]}]}", surf, lp, mp, CW, CHh / 2, surf, lp, mp, CHh / 2, CW, CHh - CHh / 2);
        FILE *f = fopen(sj, "w"); fputs(js, f); fclose(f);
        inksplat_cfg c7 = c; c7.out = o7; c7.coarse = 0; c7.face = 1; c7.face_a = -3; c7.face_b = 0; c7.nexcl = 1;
        const int64_t bx[6] = {0, 0, 0, NV, NV, NV / 3}; memcpy(c7.excl[0], bx, sizeof bx);
        c7.crop[0] = 0; c7.crop[1] = 0; c7.crop[2] = CW; c7.crop[3] = CHh / 2; CHECK(inksplat_run(&c7) == 0, "face crop 1");
        c7.crop[1] = CHh / 2; c7.crop[3] = CHh - CHh / 2; c7.merge = 1; c7.name = "seg_b"; CHECK(inksplat_run(&c7) == 0, "face crop 2");
        store *s7 = store_open(o7); z3 *ref = z3_open(s7, "1", nullptr); uint8_t *rb = malloc((size_t)NV * NV * NV);
        CHECK(ref && !z3_read(ref, z0, nn, rb, 4), "read face2");
        store *cs = store_open(dir); z3 *ct0 = z3_open(cs, "ct.zarr/1", nullptr);
        inksplat_cfg c8 = c7; c8.crop[2] = 0; c8.merge = 0; c8.level = 0;
        char dcd[600]; snprintf(dcd, sizeof dcd, "%s/dcache", dir);
        inkfly_gpu_config(-1);   /* opens 0, 1: the CPU code; open 2: the GPU (nn_inkfly.cu), memory cache only, when a device answers */
        for (int open = 0; open < 3; open++) {   /* open 0: computed (cold), then from memory; open 1: a fresh array over the same disk cache */
            if (open == 2) { if (!getenv("UFSM_TEST_NO_GPU")) inkfly_gpu_config(0); else break; }
            z3 *ctx = open ? z3_open(cs, "ct.zarr/1", nullptr) : ct0;
            dcache_stats d0; dcache_get_stats(&d0);
            const uint64_t g0 = inkfly_gpu_chunks();
            z3 *fly = ctx ? inkfly_open(ctx, sj, &c8, 64, open == 2 ? nullptr : dcd, nullptr) : nullptr; CHECK(fly != nullptr, "inkfly_open %d", open);
            if (!fly) break;
            struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
            for (int pass = 0; pass < 2; pass++) {   /* odd boxes */
                const int64_t bo[3] = {37, 11, 5}, bn[3] = {NV - 60, NV - 20, NV - 9}; uint8_t *fb = malloc((size_t)bn[0] * bn[1] * bn[2]);
                CHECK(!z3_read(fly, bo, bn, fb, 4), "read onfly");
                if (open == 2) { uint64_t h = 1469598103934665603ull; for (size_t q = 0; q < (size_t)bn[0] * bn[1] * bn[2]; q++) h = (h ^ fb[q]) * 1099511628211ull;
                                 fprintf(stderr, "onfly gpu pass %d: bytes fnv %016llx\n", pass, (unsigned long long)h); }
                long diff = 0, kn = 0;
                for (int64_t z = 0; z < bn[0]; z++) for (int64_t y = 0; y < bn[1]; y++) for (int64_t x = 0; x < bn[2]; x++) {
                    const uint8_t a = fb[(z * bn[1] + y) * bn[2] + x], b = rb[((z + bo[0]) * NV + y + bo[1]) * NV + x + bo[2]]; diff += a != b; kn += b != 255; }
                fprintf(stderr, "onfly open %d pass %d: %ld of %ld voxels differ from the stored splat (%ld known)\n", open, pass, diff, (long)(bn[0] * bn[1] * bn[2]), kn);
                /* exact under UFSM_STRICT_MATH. Under fast math the on-the-fly target (inkfly_chunk, a separate copy of the splat /
                   face search; open 2: nn_inkfly.cu) and inksplat_run round their voxel picks (floor(x + 0.5)), face search and
                   ceil(1 / hpx) apart at boundaries, and a whole voxel moves: <= 5e-3 of the known voxels may differ (tests/fm_tol.h) */
                CHECK(diff <= (long)fm_allow((size_t)kn, 5e-3) && kn > 10000, "onfly == stored, open %d pass %d: %ld differ, %ld known", open, pass, diff, kn);
                free(fb);
            }
            { struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1); fprintf(stderr, "onfly open %d: reads %.3f s\n", open, (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec)); }
            dcache_stats d1; dcache_get_stats(&d1);
            fprintf(stderr, "dcache open %d: %llu computed, %llu from disk, %llu from memory\n", open, (unsigned long long)(d1.cells_computed - d0.cells_computed),
                    (unsigned long long)(d1.cells_disk - d0.cells_disk), (unsigned long long)(d1.cells_mem - d0.cells_mem));
            if (!open) CHECK(d1.cells_computed > d0.cells_computed && d1.cells_mem > d0.cells_mem && d1.cells_disk == d0.cells_disk, "dcache: cold computes, warm from memory");
            else if (open == 1) CHECK(d1.cells_computed == d0.cells_computed && d1.cells_disk > d0.cells_disk, "dcache: a reopened array reads its cells from disk");
            else {   /* a visible device that computed nothing is a failure; no device at all (a CPU run) a skip of this part */
                int ndev = 0; if (cudaGetDeviceCount(&ndev) != cudaSuccess) ndev = 0;
                const unsigned long long gc = (unsigned long long)(inkfly_gpu_chunks() - g0);
                const unsigned long long ov = inkfly_gpu_over ? inkfly_gpu_over() : 0;
                fprintf(stderr, "onfly on the GPU: %llu chunks, %llu over the budget%s\n", gc, ov, gc || ov ? "" : ndev ? " with a device visible (FAIL)" : " (no device: the CPU ran; skipped)");
                if (getenv("TIS_OVER"))   /* (UFSM_INKFLY_GPU_MB=1: nothing fits; every chunk on the CPU, the same bytes, the GPU path left on) */
                    CHECK(ndev == 0 || (gc == 0 && ov > 0), "onfly over the budget: %llu GPU chunks, %llu over", gc, ov);
                else CHECK(gc > 0 || ndev == 0, "onfly on the GPU: %d device(s) visible, no chunk computed on one", ndev);
            }
            if (open == 0) {   /* the MIL rays (inkfly_rays) of the whole volume: every ray's point inside, mode 1 (no face) a subset of mode 0 */
                void *hd = nullptr; z3 *f2 = inkfly_open(z3_open(cs, "ct.zarr/1", nullptr), sj, &c8, 64, nullptr, &hd);
                const int64_t wo[3] = {0, 0, 0}, wn[3] = {NV, NV, NV}; inkray *r0 = nullptr, *r1 = nullptr; size_t n0 = 0, n1 = 0;
                CHECK(f2 && !inkfly_rays(hd, wo, wn, 0, 1, 4, &r0, &n0) && !inkfly_rays(hd, wo, wn, 1, 1, 4, &r1, &n1), "inkfly_rays");
                long inside = 0, ink = 0; for (size_t i = 0; i < n0; i++) { int in = 1; for (int d = 0; d < 3; d++) in &= r0[i].p[d] >= -0.5f && r0[i].p[d] < NV; inside += in; ink += r0[i].y; }
                fprintf(stderr, "mil rays: %zu (all supervised pixels, %ld ink), %zu without a face\n", n0, ink, n1);
                CHECK(n0 > 1000 && inside == (long)n0 && n1 < n0, "rays: %zu, %ld inside, %zu without a face", n0, inside, n1);
                {   /* the face search in z slabs (16-row slabs: the box in many reads) == the whole box in one: the same rays, byte for byte */
                    setenv("UFSM_INKRAY_SLAB_MB", "0.001", 1); inkray *r2 = nullptr; size_t n2 = 0;
                    CHECK(!inkfly_rays(hd, wo, wn, 1, 1, 4, &r2, &n2), "inkfly_rays slabbed");
                    CHECK(n2 == n1 && (!n1 || !memcmp(r1, r2, n1 * sizeof *r1)), "rays in z slabs: %zu vs %zu whole-box, %s", n2, n1, n2 == n1 && (!n1 || !memcmp(r1, r2, n1 * sizeof *r1)) ? "identical" : "DIFFER");
                    unsetenv("UFSM_INKRAY_SLAB_MB"); free(r2);
                }
                {   /* the face search on the GPU (inkfly_gpu_rayface, 2026-11-20) == the CPU's rays byte for byte, whole box and in z slabs; with a
                       device visible the rays were faced there (TIS_OVER: the budget refuses them, the CPU loop gives the same bytes) */
                    int ndev = 0; if (cudaGetDeviceCount(&ndev) != cudaSuccess) ndev = 0;
                    if (ndev && !getenv("UFSM_TEST_NO_GPU") && inkfly_gpu_rays) {
                        inkfly_gpu_config(0); const unsigned long long g0r = inkfly_gpu_rays();
                        for (int sl = 0; sl < 2; sl++) {
                            if (sl) setenv("UFSM_INKRAY_SLAB_MB", "0.001", 1);
                            inkray *r3 = nullptr; size_t n3 = 0;
                            CHECK(!inkfly_rays(hd, wo, wn, 1, 1, 4, &r3, &n3), "inkfly_rays on the GPU");
                            const int same = n3 == n1 && (!n1 || !memcmp(r1, r3, n1 * sizeof *r1));
                            fprintf(stderr, "mil rays, face search on the GPU%s: %zu vs %zu on the CPU, %s\n", sl ? " (z slabs)" : "", n3, n1, same ? "identical" : "DIFFER");
                            CHECK(same, "GPU ray face search differs from the CPU's%s", sl ? " (z slabs)" : "");
                            free(r3); unsetenv("UFSM_INKRAY_SLAB_MB");
                        }
                        const unsigned long long gr = inkfly_gpu_rays() - g0r;
                        fprintf(stderr, "mil rays faced on the GPU: %llu\n", gr);
                        CHECK(getenv("TIS_OVER") ? gr == 0 : gr >= 2 * n0, "rays faced on the GPU: %llu (want %s)", gr, getenv("TIS_OVER") ? "0 over the budget" : ">= 2 x all rays");
                        inkfly_gpu_config(-1);
                    }
                }
                free(r0); free(r1); if (f2) z3_close(f2);
            }
            z3_close(fly);
        }
        {   /* the onfly "map" (2026-11-22, 1.1 um teachers from 2.4 um segments): (1) an identity reg.json == no map, byte for byte; (2) A = 2 I
               (fine = coarse / 2) on a mesh written at 2x its coordinates == the original mesh with the face distances halved by hand (the
               map's own scale, cbrt |det A^-1|); (3) the held-out boxes and the coarse key through the map */
            char mI[512], m2[512], s2x[512], sjo[512], sj2[512], js2[2048];
            snprintf(mI, sizeof mI, "%s/regI.json", dir); snprintf(m2, sizeof m2, "%s/reg2.json", dir); snprintf(s2x, sizeof s2x, "%s/seg2x.tifxyz", dir);
            snprintf(sjo, sizeof sjo, "%s/segs_o.json", dir); snprintf(sj2, sizeof sj2, "%s/segs_2x.json", dir);
            const char *rj = "{\"coarse\": {\"key\": \"c/ct2x.zarr/\", \"um\": 2}, \"fine\": {\"key\": \"ct.zarr\", \"um\": 1}, \"A\": [[%d, 0, 0, 0], [0, %d, 0, 0], [0, 0, %d, 0]]}";
            FILE *g = fopen(mI, "w"); fprintf(g, rj, 1, 1, 1); fclose(g); g = fopen(m2, "w"); fprintf(g, rj, 2, 2, 2); fclose(g);
            float *X2 = malloc(GW * GH * 4), *Y2 = malloc(GW * GH * 4), *Z2 = malloc(GW * GH * 4);
            for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) { double P[3]; surf_at(u, v, P); X2[v * GW + u] = (float)(2 * (float)P[2]); Y2[v * GW + u] = (float)(2 * (float)P[1]); Z2[v * GW + u] = (float)(2 * (float)P[0]); }
            CHECK(!tifxyz_write(s2x, GW, GH, X2, Y2, Z2, STEP, 1.0, -1, "test2x"), "tifxyz 2x"); free(X2); free(Y2); free(Z2);
            g = fopen(sjo, "w"); fprintf(g, "{\"segments\": [{\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\"}]}", surf, lp, mp); fclose(g);
            snprintf(js2, sizeof js2, "{\"segments\": [{\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\"}]}", s2x, lp, mp); g = fopen(sj2, "w"); fputs(js2, g); fclose(g);
            const size_t nv3 = (size_t)NV * NV * NV; uint8_t *buf[4]; const int64_t wo[3] = {0, 0, 0}, wn[3] = {NV, NV, NV}; long known[4] = {0};
            for (int k = 0; k < 4; k++) {   /* 0 no map, 1 identity map, 2 the original mesh at half the face distances, 3 the 2x mesh through A = 2 I */
                inksplat_cfg cm = c8; cm.nexcl = 0; cm.crop[2] = 0;
                if (k == 1) cm.map = mI;
                if (k == 2) { cm.face_a *= 0.5; cm.face_b *= 0.5; cm.face_lo *= 0.5; cm.face_hi *= 0.5; }
                if (k == 3) cm.map = m2;
                buf[k] = malloc(nv3); z3 *fz = inkfly_open(z3_open(cs, "ct.zarr/1", nullptr), k == 3 ? sj2 : sjo, &cm, 64, nullptr, nullptr);
                CHECK(fz && !z3_read(fz, wo, wn, buf[k], 4), "map test: onfly %d", k);
                for (size_t q = 0; q < nv3; q++) known[k] += buf[k][q] != 255;
                if (fz) z3_close(fz);
            }
            const int s01 = !memcmp(buf[0], buf[1], nv3), s23 = !memcmp(buf[2], buf[3], nv3);
            fprintf(stderr, "onfly map: identity %s (%ld / %ld known), A = 2 I vs the hand-halved splat %s (%ld / %ld known)\n", s01 ? "identical" : "DIFFERS", known[1], known[0],
                    s23 ? "identical" : "DIFFERS", known[3], known[2]);
            CHECK(s01 && known[0] > 10000, "onfly map: the identity map differs from no map (%ld known)", known[0]);
            CHECK(s23 && known[2] > 10000, "onfly map: A = 2 I differs from the hand splat (%ld known)", known[2]);
            for (int k = 0; k < 4; k++) free(buf[k]);
            int64_t hb[1][6] = {{10, 20, 30, 5, 6, 7}}; char ck[512] = ""; double cu = 0;
            CHECK(!inkfly_map_boxes(m2, hb, 1, 0) && hb[0][0] == 5 && hb[0][1] == 10 && hb[0][2] == 15 && hb[0][3] == 3 && hb[0][4] == 3 && hb[0][5] == 4,
                  "map boxes: [%lld %lld %lld %lld %lld %lld], want [5 10 15 3 3 4]", (long long)hb[0][0], (long long)hb[0][1], (long long)hb[0][2], (long long)hb[0][3], (long long)hb[0][4], (long long)hb[0][5]);
            CHECK(!inkfly_map_coarse(m2, ck, sizeof ck, &cu) && !strcmp(ck, "c/ct2x.zarr/") && cu == 2, "map coarse: %s %g", ck, cu);
        }
        free(rb); if (ref) z3_close(ref); store_close(s7); store_close(cs);
    }
    /* validation rasters (per-pixel holdouts): the stored splat with --val (exclude, then only) == the on-the-fly target with the segment's
       "validation", and exclude + only together cover the plain splat's known voxels */
    {
        uint8_t *vr = calloc((size_t)CW * CHh, 1); for (int r = 0; r < CHh / 3; r++) memset(vr + (size_t)r * CW, 1, CW);   /* (0/1: the 0/1 rule) */
        char vp[512], sj[512], js[2048]; snprintf(vp, sizeof vp, "%s/validation.tif", dir); tiffw_write(vp, CW, CHh, 1, vr, 1); free(vr);
        snprintf(sj, sizeof sj, "%s/segs_val.json", dir);
        snprintf(js, sizeof js, "{\"segments\": [{\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\", \"validation\": \"%s\"}]}", surf, lp, mp, vp);
        FILE *f = fopen(sj, "w"); fputs(js, f); fclose(f);
        long kn[2] = {0, 0};
        for (int mode = 1; mode <= 2; mode++) {
            char ov[512]; snprintf(ov, sizeof ov, "%s/val%d.zarr", dir, mode);
            inksplat_cfg cv = c; cv.out = ov; cv.coarse = 0; cv.face = 1; cv.face_a = -3; cv.face_b = 0; cv.val = vp; cv.val_mode = mode;
            CHECK(inksplat_run(&cv) == 0, "val splat %d", mode);
            store *sv = store_open(ov); z3 *ref = z3_open(sv, "1", nullptr); uint8_t *rb = malloc((size_t)NV * NV * NV), *fb = malloc((size_t)NV * NV * NV);
            CHECK(ref && !z3_read(ref, z0, nn, rb, 4), "read val splat");
            store *cs = store_open(dir); inksplat_cfg c9 = cv; c9.level = 0;
            z3 *fly = inkfly_open(z3_open(cs, "ct.zarr/1", nullptr), sj, &c9, 64, nullptr, nullptr); CHECK(fly && !z3_read(fly, z0, nn, fb, 4), "val onfly %d", mode);
            long diff = 0; for (size_t k = 0; k < (size_t)NV * NV * NV; k++) { diff += rb[k] != fb[k]; kn[mode - 1] += rb[k] != 255; }
            fprintf(stderr, "validation %s: %ld known, onfly vs stored: %ld differ\n", mode == 1 ? "exclude" : "only", kn[mode - 1], diff);
            CHECK(diff <= (long)fm_allow((size_t)kn[mode - 1], 5e-3) && kn[mode - 1] > 1000, "validation %d: %ld differ, %ld known", mode, diff, kn[mode - 1]);   /* (the bound above) */
            if (fly) z3_close(fly);
            free(rb); free(fb); if (ref) z3_close(ref); store_close(sv); store_close(cs);
        }
    }
    /* --exclude: a held-out box over the low-x half: nothing written there, the rest as before */
    {
        char o5[512]; snprintf(o5, sizeof o5, "%s/excl.zarr", dir);
        inksplat_cfg c5 = c; c5.out = o5; c5.coarse = 0; c5.nexcl = 1;
        const int64_t bx[6] = {0, 0, 0, NV, NV, NV / 2}; memcpy(c5.excl[0], bx, sizeof bx);
        CHECK(inksplat_run(&c5) == 0, "exclude splat");
        store *s5 = store_open(o5); z3 *z = z3_open(s5, "1", nullptr);
        CHECK(z && !z3_read(z, z0, nn, ob, 4), "read exclude");
        long win = 0, wout = 0;
        for (int64_t k = 0; k < (int64_t)NV * NV * NV; k++) if (ob[k] != 255) { if (k % NV < NV / 2) win++; else wout++; }
        fprintf(stderr, "exclude: written %ld inside the box, %ld outside\n", win, wout);
        CHECK(win == 0 && wout > 10000, "exclude box: %ld inside, %ld outside", win, wout);
        if (z) z3_close(z);
        store_close(s5);
    }
    /* merge: a second splat with all-ink labels into the first output: ink wins (max) where both are known */
    {
        for (size_t k = 0; k < (size_t)CW * CHh; k++) lab[k] = 255;
        char l2[512]; snprintf(l2, sizeof l2, "%s/allink.tif", dir); tiffw_write(l2, CW, CHh, 1, lab, 1);
        inksplat_cfg c3 = c; c3.labels = l2; c3.merge = 1; c3.name = "seg2"; c3.coarse = 0;
        CHECK(inksplat_run(&c3) == 0, "merge splat");
        z3 *z = z3_open(os, "1", nullptr); CHECK(z && !z3_read(z, z0, nn, ob, 4), "read merged");
        long n0 = 0; for (size_t k = 0; k < (size_t)NV * NV * NV; k++) n0 += ob[k] == 0;
        CHECK(n0 == 0, "merged no-ink voxels left: %ld", n0);
        z3_close(z);
    }
    /* 3D point labels: an ink ball on papyrus, a no-ink ball on papyrus, an ink ball in void (unknown) */
    {
        double P[3]; surf_at(10, 10, P); double Q[3]; surf_at(20, 12, Q);
        char js[512], pj[512], o4[512]; snprintf(pj, sizeof pj, "%s/pts.json", dir); snprintf(o4, sizeof o4, "%s/pts.zarr", dir);
        snprintf(js, sizeof js, "{\"collections\": {\"1\": {\"name\": \"ink\", \"points\": {\"1\": {\"p\": [%g, %g, %g]}, \"2\": {\"p\": [200, 200, 20]}}},"
                 " \"2\": {\"name\": \"noink\", \"points\": {\"1\": {\"p\": [%g, %g, %g]}}}}}", P[2], P[1], P[0], Q[2], Q[1], Q[0]);
        FILE *f = fopen(pj, "w"); fputs(js, f); fclose(f);
        char *av[] = {"ufsm", "ink-splat", o4, "--points", pj, "--radius-um", "2", "--ct", dir, "ct.zarr", "--ct-um", "1", "--iso", "100", "--coarse", "0"};
        CHECK(cmd_ink_splat(16, av) == 0, "points");
        store *s4 = store_open(o4); z3 *z = z3_open(s4, "1", nullptr); CHECK(z && !z3_read(z, z0, nn, ob, 4), "read points");
        const int64_t a = (int64_t)floor(P[0] + 0.5), b = (int64_t)floor(Q[0] + 0.5);
        CHECK(ob[(a * NV + (int64_t)floor(P[1] + 0.5)) * NV + (int64_t)floor(P[2] + 0.5)] == 254, "ink point");
        CHECK(ob[(b * NV + (int64_t)floor(Q[1] + 0.5)) * NV + (int64_t)floor(Q[2] + 0.5)] == 0, "no-ink point");
        CHECK(ob[(20 * NV + 200) * NV + 200] == 255, "void point unknown");
        long k254 = 0; for (size_t k = 0; k < (size_t)NV * NV * NV; k++) k254 += ob[k] == 254;
        CHECK(k254 >= 20 && k254 <= 40, "ink ball voxels %ld (radius 2: 33)", k254);
        if (z) z3_close(z);
    }
    snprintf(p, sizeof p, "%s/out.zarr/segments/seg.tifxyz.json", dir);
    CHECK(access(p, R_OK) == 0, "summary %s", p);
    {   /* ink-score on the canvas labels: the labels themselves as the map score F1 1 / AUC 1, the inverted labels AUC 0, a constant map
           AUC 0.5; with the mask, the masked rows count for nothing (the same perfect score) */
        const int CW = (int)(GW * STEP), CHh = (int)(GH * STEP); uint8_t *m = malloc((size_t)CW * CHh); double f1 = 0, auc = 0;
        char mp[512], kp[512]; snprintf(mp, sizeof mp, "%s/map.tif", dir); snprintf(kp, sizeof kp, "%s/mask.tif", dir);
        for (int r = 0; r < CHh; r++) for (int c = 0; c < CW; c++) m[r * CW + c] = ink_at(r, c) ? 230 : 20;
        tiffw_write(mp, CW, CHh, 1, m, 1);
        CHECK(!ink_score(dir, mp, nullptr, &f1, &auc) && f1 > 0.97 && auc > 0.98, "ink-score labels as the map: F1 %.4f AUC %.4f (want ~1)", f1, auc);
        CHECK(!ink_score(dir, mp, kp, &f1, &auc) && f1 > 0.97 && auc > 0.98, "ink-score with the mask: F1 %.4f AUC %.4f", f1, auc);
        for (size_t k = 0; k < (size_t)CW * CHh; k++) m[k] = 250 - m[k];
        tiffw_write(mp, CW, CHh, 1, m, 1);
        CHECK(!ink_score(dir, mp, nullptr, &f1, &auc) && auc < 0.03, "ink-score inverted map: AUC %.4f (want ~0)", auc);
        memset(m, 128, (size_t)CW * CHh); tiffw_write(mp, CW, CHh, 1, m, 1);
        CHECK(!ink_score(dir, mp, nullptr, &f1, &auc) && fabs(auc - 0.5) < 1e-6, "ink-score constant map: AUC %.6f (want 0.5)", auc);
        fprintf(stderr, "ink-score: perfect / masked / inverted / constant maps scored as expected\n");
        free(m);
    }
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {}
    if (fails) { fprintf(stderr, "test_ink_splat: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_ink_splat: ok\n");
    return 0;
}
