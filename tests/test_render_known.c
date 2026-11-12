/* ufsm render --known (label zarrs, e.g. ink-splat output: 0..254 values, 255 = unknown): per pixel the max over the known values of
   the nearest voxels along the normal within +-depth, 0 when none is known. A lossless label volume (mostly 255, a known shell of
   values 0..254 around a cylinder, a hole of unknown in it) through a curved grid with invalid vertices, by the ufsm binary
   (argv[1]) on both CPU paths (the box read once; per tile with UFSM_RENDER_MEM tiny), vs the same arithmetic here on the dense
   volume. Nearest-voxel rounding sits on exact .5 boundaries for a few pixels, so up to 0.2% of the pixels may differ between the two
   builds' floating point; the plain (trilinear) render of the same volume must differ (the 255s win there), and --known with --fill
   or --joint is refused. */
#include "imgops.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void ref_known(const float *X, const float *Y, const float *Z, int w, int h, double gs, const uint8_t *vol, const int64_t n[3], int depth, uint8_t *img, int ow, int oh) {
    for (int py = 0; py < oh; py++) for (int px = 0; px < ow; px++) {
        const double gu = px / gs, gv = py / gs;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = gu - c0, fv = gv - r0;
        const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
        if (!(Z[i00] >= 0 && Z[i01] >= 0 && Z[i10] >= 0 && Z[i11] >= 0)) continue;
        const float *A[3] = {Z, Y, X}; double P[3], du[3], dv[3];
        for (int d = 0; d < 3; d++) {
            const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
            P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
            du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
        }
        double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
        for (int d = 0; d < 3; d++) N[d] /= nl;
        int m = 0;
        for (int t = -depth; t <= depth; t++) {
            const int64_t q0 = (int64_t)floor(P[0] + t * N[0] + 0.5), q1 = (int64_t)floor(P[1] + t * N[1] + 0.5), q2 = (int64_t)floor(P[2] + t * N[2] + 0.5);
            if (q0 < 0 || q1 < 0 || q2 < 0 || q0 >= n[0] || q1 >= n[1] || q2 >= n[2]) continue;
            const uint8_t b = vol[((size_t)q0 * n[1] + q1) * n[2] + q2]; if (b != 255 && b > m) m = b;
        }
        img[(size_t)py * ow + px] = (uint8_t)m;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_render_known <ufsm>\n"); return 2; }
    char dir[] = "/tmp/test_render_known_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char p[1200], cmd[4000];
    const int S = 128; const int64_t shape[3] = {256, 256, 256};
    uint8_t *vol = malloc((size_t)256 * 256 * 256), *buf = malloc((size_t)S * S * S);
    for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) {
        const double rr = hypot(x - 128.0, y - 126.0); int v = 255;   /* unknown but for a shell 3 voxels thick at radius 88..91 */
        if (rr >= 88 && rr < 91) v = (x * 7 + y * 3 + z * 5) % 255;
        if (rr >= 88 && rr < 91 && z > 100 && z < 120 && x > 128 && y > 150) v = 255;   /* a hole of unknown in the shell */
        if (rr >= 91 && rr < 92 && (z / 8) % 2) v = 0;   /* known no-ink beside it */
        vol[((size_t)z * 256 + y) * 256 + x] = (uint8_t)v;
    }
    snprintf(p, sizeof p, "%s/lab.zarr/0", dir);
    z3w *zw = z3w_create(p, shape, S, 0, 255, "{}"); CHECK(zw != nullptr, "z3w_create %s", z3w_error());
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) memcpy(buf + ((size_t)z * S + y) * S, vol + ((size_t)(sz * S + z) * 256 + sy * S + y) * 256 + sx * S, S);
        CHECK(!z3w_write_shard(zw, sz, sy, sx, buf, 2), "write shard %s", z3w_error());
    }
    CHECK(zw && !z3w_close(zw), "close");
    {   /* the store reads back what the reference uses */
        z3 *z = z3_open(store_open(dir), "lab.zarr/0", nullptr); CHECK(z != nullptr, "z3_open %s", z3_error());
        uint8_t *d = malloc((size_t)256 * 256 * 256); const int64_t o[3] = {0, 0, 0};
        CHECK(z && !z3_read(z, o, shape, d, 2) && !memcmp(d, vol, (size_t)256 * 256 * 256), "label volume round trip"); free(d); if (z) z3_close(z);
    }
    /* a cylinder section at radius 89.5 +- 2 (in and out of the shell), rows along z; a hole of invalid vertices */
    const int w = 60, h = 50; float *X = malloc(sizeof(float) * w * h), *Y = malloc(sizeof(float) * w * h), *Z = malloc(sizeof(float) * w * h);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; const double th = 0.2 + c * 0.05, rad = 89.5 + 2 * sin(r * 0.21 + c * 0.13);
        X[i] = (float)(128 + rad * cos(th)); Y[i] = (float)(126 + rad * sin(th)); Z[i] = (float)(12 + r * 4.6);
        if (r > 30 && r < 35 && c > 8 && c < 15) X[i] = Y[i] = Z[i] = -1;
    }
    snprintf(p, sizeof p, "mkdir -p '%s/wraps' '%s/o1' '%s/o2' '%s/o3'", dir, dir, dir, dir); CHECK(!system(p), "mkdir");
    snprintf(p, sizeof p, "%s/wraps/w000", dir); CHECK(!tifxyz_write(p, w, h, X, Y, Z, 1.0, 8.0, 0, "test_render_known"), "tifxyz_write");
    const double gs = 2.5; const int depth = 3, ow = render_out_px(w, gs), oh = render_out_px(h, gs); const size_t no = (size_t)ow * oh;
    uint8_t *ref = calloc(no, 1); ref_known(X, Y, Z, w, h, gs, vol, shape, depth, ref, ow, oh);
    size_t nk = 0, n255 = 0; for (size_t i = 0; i < no; i++) { nk += ref[i] != 0; n255 += ref[i] == 255; }
    CHECK(nk > no / 3 && nk < no && !n255, "reference: %zu of %zu pixels known and > 0, %zu at 255", nk, no, n255);
    const char *env[2] = {"", "UFSM_RENDER_MEM=0.00001 "}, *sub[2] = {"o1", "o2"};
    for (int m = 0; m < 2; m++) {
        snprintf(cmd, sizeof cmd, "%s'%s' render '%s/wraps' '%s/%s' --volume '%s' lab.zarr/0 --known --depth %d --scale %g --tile 64 --threads 3 2> '%s/%s.log'", env[m], argv[1], dir, dir, sub[m], dir, depth, gs, dir, sub[m]);
        CHECK(!system(cmd), "render --known (path %d) failed", m);
        snprintf(p, sizeof p, "%s/%s/w000.tif", dir, sub[m]); int w2, h2; uint8_t *im = img_read_u8(p, &w2, &h2);
        CHECK(im && w2 == ow && h2 == oh, "path %d: output %dx%d, expected %dx%d", m, w2, h2, ow, oh);
        size_t bad = 0, s255 = 0; for (size_t i = 0; im && w2 == ow && h2 == oh && i < no; i++) { bad += im[i] != ref[i]; s255 += im[i] == 255; }
        CHECK(bad <= no / 500 && !s255, "path %d: %zu of %zu pixels differ from the reference, %zu at 255", m, bad, no, s255);
        fprintf(stderr, "test_render_known: path %d (%s): %zu of %zu pixels differ\n", m, m ? "per tile" : "box read once", bad, no);
        free(im);
    }
    {   /* the plain render of the label volume is another picture (unknown 255 wins the max) */
        snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/o3' --volume '%s' lab.zarr/0 --depth %d --scale %g --gpu 0 --threads 3 2> '%s/o3.log'", argv[1], dir, dir, dir, depth, gs, dir);
        CHECK(!system(cmd), "plain render failed");
        snprintf(p, sizeof p, "%s/o3/w000.tif", dir); int w2, h2; uint8_t *im = img_read_u8(p, &w2, &h2); size_t df = 0;
        for (size_t i = 0; im && w2 == ow && h2 == oh && i < no; i++) df += im[i] != ref[i];
        CHECK(im && df > no / 3, "plain render: only %zu of %zu pixels differ from --known", df, no); free(im);
    }
    snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/o3' --volume '%s' lab.zarr/0 --known --joint 1 2> /dev/null", argv[1], dir, dir, dir);
    CHECK(system(cmd) != 0, "--known --joint 1 accepted");
    snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/o3' --volume '%s' lab.zarr/0 --ct '%s' lab.zarr/0 --known --fill 2 2> /dev/null", argv[1], dir, dir, dir, dir);
    CHECK(system(cmd) != 0, "--known --fill accepted");
    free(vol); free(buf); free(ref); free(X); free(Y); free(Z);
    if (!fails) { snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {} }
    if (fails) { fprintf(stderr, "test_render_known: %d failures (kept %s)\n", fails, dir); return 1; }
    printf("test_render_known: OK\n");
    return 0;
}
