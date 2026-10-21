/* ufsm ink-splat round trip: a synthetic curved surface (tifxyz), a 2D ink label + supervision mask on its canvas and a CT with a
   papyrus slab around the surface and a void hole. Splat, then sample the written zarr back along each pixel's normal and recover
   the label: ink 254 / no ink 0 inside the band on papyrus, unknown (255) outside the band, in the hole, and at unsupervised pixels;
   --flip moves a one-sided band to the other side; the coarse level keeps the band (known-max); --merge combines by max. */
#include "inksplat.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    snprintf(p, sizeof p, "%s/out.zarr/segments/seg.tifxyz.json", dir);
    CHECK(access(p, R_OK) == 0, "summary %s", p);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {}
    if (fails) { fprintf(stderr, "test_ink_splat: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_ink_splat: ok\n");
    return 0;
}
