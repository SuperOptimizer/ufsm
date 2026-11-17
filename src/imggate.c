/* ufsm img-gate IMG.tif GATE.tif [GATE.tif ..] [--out OUT.tif]: a uint8 render multiplied by the max of its gate renders / 255 (the
   strip's papyrus gate, scripts/strip/strip_refill.sh: the ink render x max(c12, c13) / 255), byte for byte numpy's float32
   (i.astype(f32) * (max(g).astype(f32) / 255) + 0.5).astype(u8) that it replaces: one 256 x 256 table built with float32
   rounding forced by the casts (a double product of a byte and a float is exact; the casts round once, as numpy does), so
   the fast-math flags cannot change it (tests/test_img_gate.py checks all 65536 pairs against numpy).
   OUT defaults to IMG (written to OUT.tmpPID, renamed). Uncompressed, like the numpy step. 0.5 GB stripes: ~5 s (the deflate reads) instead of
   28-180 s for the numpy step (float32 temporaries of 2 GB each; 0332 stripimg 2026-11-17). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "imgops.h"
#include "optcheck.h"
#include "tiffw.h"

void img_gate_lut(uint8_t lut[256][256]) {   /* lut[i][g] */
    for (int g = 0; g < 256; g++) {
        volatile float p = (float)((double)g / 255.0);   /* (float32 division: the double quotient rounded once to float; exact for all 256 per the test) */
        for (int i = 0; i < 256; i++) { volatile float q = (float)((double)i * (double)p); volatile float v = (float)((double)q + 0.5); lut[i][g] = (uint8_t)(int)v; }
    }
}

int cmd_img_gate(int argc, char **argv) {
    const char *out = nullptr; const char *in[16]; int n = 0;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], opt_n("--out")) && a + 1 < argc) { out = argv[++a]; continue; }
        if (n >= 16) { fprintf(stderr, "img-gate: at most 15 gates\n"); return 2; }
        in[n++] = argv[a];
    }
    if (n < 2) { fprintf(stderr, "usage: ufsm img-gate IMG.tif GATE.tif [GATE.tif ..] [--out OUT.tif]: IMG x max(GATEs) / 255 (numpy float32 rounding)\n"); return 2; }
    if (!out) out = in[0];
    int W, H; uint8_t *im = img_read_u8(in[0], &W, &H), *g = nullptr;
    if (!im) { fprintf(stderr, "img-gate: cannot read %s\n", in[0]); return 1; }
    const size_t np = (size_t)W * H;
    for (int k = 1; k < n; k++) {
        int w, h; uint8_t *x = img_read_u8(in[k], &w, &h);
        if (!x || w != W || h != H) { fprintf(stderr, "img-gate: cannot read %s as a %d x %d image\n", in[k], W, H); free(x); free(im); free(g); return 1; }
        if (!g) { g = x; continue; }
        for (size_t i = 0; i < np; i++) if (x[i] > g[i]) g[i] = x[i];
        free(x);
    }
    static uint8_t lut[256][256]; img_gate_lut(lut);   /* mgpu: main (the img-gate command, one thread) */
    for (size_t i = 0; i < np; i++) im[i] = lut[im[i]][g[i]];
    free(g);
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp%d", out, (int)getpid());
    if (tiffw_write(tmp, W, H, 1, im, 0) || rename(tmp, out)) { fprintf(stderr, "img-gate: cannot write %s\n", out); unlink(tmp); free(im); return 1; }
    free(im);
    return 0;
}
