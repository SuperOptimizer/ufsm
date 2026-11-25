/* ufsm img-gate IMG.tif GATE.tif [GATE.tif ..] [--out OUT.tif] [--min T]: a uint8 render multiplied by the max of its gate renders / 255 (the
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
#include <math.h>
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
    const char *out = nullptr; const char *in[16]; int n = 0, gmin = -1;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], opt_n("--out")) && a + 1 < argc) { out = argv[++a]; continue; }
        if (!strcmp(argv[a], opt_n("--min")) && a + 1 < argc) { gmin = opt_atoi("--min", argv[++a]); continue; }
        if (n >= 16) { fprintf(stderr, "img-gate: at most 15 gates\n"); return 2; }
        in[n++] = argv[a];
    }
    if (n < 2 || gmin > 255) { fprintf(stderr, "usage: ufsm img-gate IMG.tif GATE.tif [GATE.tif ..] [--out OUT.tif] [--min T]: IMG x max(GATEs) / 255 (numpy float32 rounding)\n"
                                            "  --min T: the gate binary, 255 where max(GATEs) >= T else 0 (IMG kept or zeroed; r3ink.sh: the filled CT render at the papyrus level)\n"); return 2; }
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
    if (gmin >= 0) for (size_t i = 0; i < np; i++) g[i] = g[i] >= gmin ? 255 : 0;   /* lut[i][255] == i: kept or zeroed */
    static uint8_t lut[256][256]; img_gate_lut(lut);   /* mgpu: main (the img-gate command, one thread) */
    for (size_t i = 0; i < np; i++) im[i] = lut[im[i]][g[i]];
    free(g);
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp%d", out, (int)getpid());
    if (tiffw_write(tmp, W, H, 1, im, 0) || rename(tmp, out)) { fprintf(stderr, "img-gate: cannot write %s\n", out); unlink(tmp); free(im); return 1; }
    free(im);
    return 0;
}

/* ufsm img-stretch CTL CT1.tif INK1.tif [CT2.tif INK2.tif ..]: strip_refill.sh's gated contrast stretch "ca,cb,ia,ib", byte for byte its
   numpy step: per stripe c = ct[ct > 0][::7], k = ink[ct >= CTL][::7] (every 7th in C order, restarting per stripe), over all stripes
   (int(percentile(c, 0.5)), max(int(percentile(c, 99.8)), lo + 1)) and the same for k at 1 / 99.9; numpy's 'linear' percentile
   (virtual index (n - 1) q, q = p / 100, its two-sided lerp) on histograms. The numpy step read and masked three 0.5 G-pixel stripes
   with Python temporaries: ~2 min of stripimg (0332 2026-11-17); this ~5 s (the deflate reads). volatile: no fused multiply-add. */
static int pct_u8(const uint64_t h[256], uint64_t n, double p) {
    if (!n) return 0;
    volatile double q = p / 100.0; volatile double vi = (double)(int64_t)(n - 1) * q;
    int64_t prev, next;
    if (vi >= (double)(n - 1)) prev = next = (int64_t)n - 1; else if (vi < 0) prev = next = 0; else { prev = (int64_t)floor(vi); next = prev + 1; }
    volatile double gam = vi - (double)prev;
    int a = -1, b = -1; uint64_t c = 0;
    for (int v = 0; v < 256 && b < 0; v++) { const uint64_t c2 = c + h[v]; if (a < 0 && (uint64_t)prev < c2) a = v; if ((uint64_t)next < c2) b = v; c = c2; }
    if (b < 0) b = a;
    const double diff = (double)(uint8_t)(b - a);
    double r;
    if (gam >= 0.5) { volatile double m = diff * (1 - gam); r = (double)b - m; } else { volatile double m = diff * gam; r = (double)a + m; }
    return (int)r;
}
int cmd_img_stretch(int argc, char **argv) {
    if (argc < 5 || (argc - 3) % 2) { fprintf(stderr, "usage: ufsm img-stretch CTL CT1.tif INK1.tif [CT2.tif INK2.tif ..]: the gated stretch ca,cb,ia,ib (strip_refill.sh)\n"); return 2; }
    const int ctl = atoi(argv[2]); uint64_t hc[256] = {0}, hk[256] = {0}, nc = 0, nk = 0;
    for (int a = 3; a + 1 < argc; a += 2) {
        int w, h, w2, h2; uint8_t *ct = img_read_u8(argv[a], &w, &h), *ik = img_read_u8(argv[a + 1], &w2, &h2);
        if (!ct || !ik || w != w2 || h != h2) { fprintf(stderr, "img-stretch: cannot read %s / %s as one size\n", argv[a], argv[a + 1]); free(ct); free(ik); return 1; }
        uint64_t jc = 0, jk = 0; const size_t np = (size_t)w * h;
        for (size_t i = 0; i < np; i++) {
            const uint8_t v = ct[i];
            if (v > 0) { if (jc++ % 7 == 0) { hc[v]++; nc++; } }
            if (v >= ctl) { if (jk++ % 7 == 0) { hk[ik[i]]++; nk++; } }
        }
        free(ct); free(ik);
    }
    const int ca = pct_u8(hc, nc, 0.5), cb0 = pct_u8(hc, nc, 99.8), ia = pct_u8(hk, nk, 1), ib0 = pct_u8(hk, nk, 99.9);
    printf("%d,%d,%d,%d\n", ca, cb0 > ca + 1 ? cb0 : ca + 1, ia, ib0 > ia + 1 ? ib0 : ia + 1);
    return 0;
}

/* ufsm img-diff A.tif B.tif: two uint8 maps of one size compared byte for byte: "max_abs N n_diff N n N" (exit 0; 1 on a size mismatch /
   unreadable). For cross-card checks of one recipe (reader3 on sm_86 vs sm_120, 2026-11-25: the lead's |d| <= 1 LSB bar). */
int cmd_img_diff(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm img-diff A.tif B.tif: max |a - b|, the count of differing bytes, the byte count\n"); return 2; }
    int w, h, w2, h2; uint8_t *a = img_read_u8(argv[2], &w, &h), *b = img_read_u8(argv[3], &w2, &h2);
    if (!a || !b || w != w2 || h != h2) { fprintf(stderr, "img-diff: cannot read %s / %s as one size\n", argv[2], argv[3]); free(a); free(b); return 1; }
    const size_t np = (size_t)w * h; size_t nd = 0; int mx = 0;
    for (size_t i = 0; i < np; i++) { const int d = abs((int)a[i] - (int)b[i]); nd += d != 0; mx = d > mx ? d : mx; }
    printf("max_abs %d n_diff %zu n %zu\n", mx, nd, np); free(a); free(b); return 0;
}

/* ufsm img-zoom IN.tif OUT.tif K: a uint8 map enlarged K x by pixel replication (nearest neighbour: no smoothing, so figure panels show the
   map's own pixels; 2026-11-25 TTA composites). Deflate TIFF. */
int cmd_img_zoom(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm img-zoom IN.tif OUT.tif K: nearest-neighbour K x enlargement of a uint8 map\n"); return 2; }
    const int k = atoi(argv[4]); int w, h; uint8_t *a = img_read_u8(argv[2], &w, &h);
    if (!a || k < 1 || k > 16) { fprintf(stderr, "img-zoom: cannot read %s, or K outside 1..16\n", argv[2]); free(a); return 1; }
    const size_t W = (size_t)w * k, H = (size_t)h * k; uint8_t *o = malloc(W * H);
    if (!o) { free(a); return 1; }
    for (size_t y = 0; y < H; y++) for (size_t x = 0; x < W; x++) o[y * W + x] = a[(y / k) * w + x / k];
    const int rc = tiffw_write(argv[3], (int)W, (int)H, 1, o, 1);
    if (rc) fprintf(stderr, "img-zoom: cannot write %s\n", argv[3]);
    free(a); free(o); return rc ? 1 : 0;
}
