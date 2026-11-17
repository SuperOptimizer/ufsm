/* ufsm render --gate (strip_refill.sh's papyrus gate in the render pass, 2026-11-17): the ink render x max(gate renders) / 255, with
   the gate volumes rendered at the volume's points (the --fill search on the CT, push-pull inpaint) in the same pass. By the ufsm
   binary (argv[1]): render --gate G1 --gate G2 must equal, byte for byte, the three separate renders (ink + CT, G1 + CT, G2 + CT, the
   same fill flags) combined by ufsm img-gate, which is the numpy float32 step strip_refill.sh used (test_img_gate.py); the CT render
   and the fill flags are those of the ungated render. argv[2] = "gpu": the gated render also on the GPU path (render --gpu 1), equal to
   the CPU's. --gate without --fill is refused. */
#include "imgops.h"
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void write_vol(const char *dir, const char *name, int kind) {   /* 256^3 in 2^3 shards of 128^3 */
    const int S = 128; const int64_t shape[3] = {256, 256, 256}; char p[1200];
    uint8_t *buf = malloc((size_t)S * S * S);
    snprintf(p, sizeof p, "%s/%s/0", dir, name);
    z3w *zw = z3w_create(p, shape, S, 0, 255, "{}"); CHECK(zw != nullptr, "z3w_create %s", z3w_error());
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
            const int gz = sz * S + z, gy = sy * S + y, gx = sx * S + x; const double rr = hypot(gx - 128.0, gy - 126.0); int v = 0;
            if (kind == 0) v = rr >= 86 && rr < 93 ? (gx * 7 + gy * 3 + gz * 5) % 256 : 0;                                    /* ink */
            else if (kind == 1) v = rr >= 87 && rr < 92 && !(gz > 100 && gz < 130 && gx > 128) ? 120 + (gx + gz) % 100 : 10;   /* CT, a void */
            else v = rr >= 85 && rr < 94 ? (kind == 2 ? (gz * 3 + gx) % 256 : (gy * 5 + gz * 2) % 256) : 0;                  /* gates */
            buf[((size_t)z * S + y) * S + x] = (uint8_t)v;
        }
        CHECK(!z3w_write_shard(zw, sz, sy, sx, buf, 2), "write shard %s", z3w_error());
    }
    CHECK(zw && !z3w_close(zw), "close");
    free(buf);
}
static uint8_t *rd(const char *dir, const char *f, int ow, int oh) {
    char p[1200]; int w, h; snprintf(p, sizeof p, "%s/%s", dir, f); uint8_t *im = img_read_u8(p, &w, &h);
    CHECK(im && w == ow && h == oh, "%s: %dx%d, expected %dx%d", f, w, h, ow, oh);
    if (im && (w != ow || h != oh)) { free(im); return nullptr; }
    return im;
}
static size_t ndiff(const uint8_t *a, const uint8_t *b, size_t n) { size_t d = 0; if (!a || !b) return n; for (size_t i = 0; i < n; i++) d += a[i] != b[i]; return d; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_render_gate <ufsm> [gpu]\n"); return 2; }
    const int gpu = argc > 2 && !strcmp(argv[2], "gpu");
    char dir[] = "/tmp/test_render_gate_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char p[1200], cmd[4000];
    write_vol(dir, "ink.zarr", 0); write_vol(dir, "ct.zarr", 1); write_vol(dir, "g1.zarr", 2); write_vol(dir, "g2.zarr", 3);
    const int w = 60, h = 50; float *X = malloc(sizeof(float) * w * h), *Y = malloc(sizeof(float) * w * h), *Z = malloc(sizeof(float) * w * h);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {   /* a cylinder section through the shell and its void; grid holes */
        const size_t i = (size_t)r * w + c; const double th = 0.2 + c * 0.05, rad = 89.5 + 3 * sin(r * 0.21 + c * 0.13);
        X[i] = (float)(128 + rad * cos(th)); Y[i] = (float)(126 + rad * sin(th)); Z[i] = (float)(12 + r * 4.6);
        if (r > 30 && r < 33 && c > 8 && c < 12) X[i] = Y[i] = Z[i] = -1;
    }
    snprintf(p, sizeof p, "mkdir -p '%s/wraps' '%s/a' '%s/g1' '%s/g2' '%s/b' '%s/c'", dir, dir, dir, dir, dir, dir); CHECK(!system(p), "mkdir");
    snprintf(p, sizeof p, "%s/wraps/w000", dir); CHECK(!tifxyz_write(p, w, h, X, Y, Z, 1.0, 8.0, 0, "test_render_gate"), "tifxyz_write");
    const double gs = 2.5; const int ow = render_out_px(w, gs), oh = render_out_px(h, gs); const size_t no = (size_t)ow * oh;
    const char *FX = "--depth 1 --fill 4 --fill-grid 50 --fill-inpaint 4 --threads 3";
    const char *vols[3] = {"ink.zarr/0", "g1.zarr/0", "g2.zarr/0"}, *outs[3] = {"a", "g1", "g2"};
    for (int k = 0; k < 3; k++) {   /* strip_refill.sh until 2026-11-17: three renders */
        snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/%s' --volume '%s' %s --ct '%s' ct.zarr/0 --scale %g %s --gpu 0 2> '%s/%s.log'", argv[1], dir, dir, outs[k], dir, vols[k], dir, gs, FX, dir, outs[k]);
        CHECK(!system(cmd), "render %s failed", vols[k]);
    }
    snprintf(cmd, sizeof cmd, "'%s' img-gate '%s/a/w000.tif' '%s/g1/w000.tif' '%s/g2/w000.tif' --out '%s/ref.tif' 2> /dev/null", argv[1], dir, dir, dir, dir);
    CHECK(!system(cmd), "img-gate failed");
    for (int m = 0; m < 1 + gpu; m++) {
        const char *o = m ? "c" : "b";
        snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/%s' --volume '%s' ink.zarr/0 --ct '%s' ct.zarr/0 --gate '%s' g1.zarr/0 --gate '%s' g2.zarr/0 --scale %g %s --gpu %d 2> '%s/%s.log'",
                 argv[1], dir, dir, o, dir, dir, dir, dir, gs, FX, m, dir, o);
        CHECK(!system(cmd), "render --gate (%s) failed", m ? "GPU" : "CPU");
        snprintf(p, sizeof p, "%s/%s", dir, o);
        uint8_t *ref = rd(dir, "ref.tif", ow, oh), *a = rd(dir, "a/w000.tif", ow, oh), *g1 = rd(dir, "g1/w000.tif", ow, oh);
        uint8_t *act = rd(dir, "a/w000_ct.tif", ow, oh), *afl = rd(dir, "a/w000_fill.tif", ow, oh);
        uint8_t *bi = rd(p, "w000.tif", ow, oh), *bc = rd(p, "w000_ct.tif", ow, oh), *bf = rd(p, "w000_fill.tif", ow, oh);
        size_t nz = 0, nfill = 0, ninp = 0, ng = 0;
        for (size_t i = 0; ref && afl && a && i < no; i++) { nz += ref[i] != 0; nfill += afl[i] > 0 && afl[i] < 50; ninp += afl[i] == RFILL_INPAINT; ng += ref[i] != a[i]; }
        CHECK(nz > no / 4 && nfill > 50 && ninp > 0 && ng > no / 4, "fixture: %zu pixels nonzero, %zu filled along the normal, %zu inpainted, %zu changed by the gate (of %zu)", nz, nfill, ninp, ng, no);
        CHECK(ndiff(g1, a, no) > no / 4, "the gate render equals the ink render");
        const size_t d0 = ndiff(bi, ref, no), d1 = ndiff(bc, act, no), d2 = ndiff(bf, afl, no);
        CHECK(!d0 && !d1 && !d2, "%s: gated ink %zu, CT %zu, fill flags %zu of %zu pixels differ from the three renders + img-gate", m ? "GPU" : "CPU", d0, d1, d2, no);
        fprintf(stderr, "test_render_gate: %s: %zu pixels, %zu gated (%zu changed), %zu filled, %zu inpainted: equal\n", m ? "GPU" : "CPU", no, nz, ng, nfill, ninp);
        if (m) { snprintf(p, sizeof p, "grep -q 'render gpu' '%s/c.log'", dir); CHECK(system(p) != 0, "the GPU render fell back to the CPU (see %s/c.log)", dir); }
        free(ref); free(a); free(g1); free(act); free(afl); free(bi); free(bc); free(bf);
    }
    snprintf(cmd, sizeof cmd, "'%s' render '%s/wraps' '%s/b' --volume '%s' ink.zarr/0 --ct '%s' ct.zarr/0 --gate '%s' g1.zarr/0 --gpu 0 2> /dev/null", argv[1], dir, dir, dir, dir, dir);
    CHECK(system(cmd) != 0, "--gate without --fill accepted");
    free(X); free(Y); free(Z);
    if (!fails) { snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {} }
    if (fails) { fprintf(stderr, "test_render_gate: %d failures (kept %s)\n", fails, dir); return 1; }
    printf("test_render_gate: OK\n");
    return 0;
}
