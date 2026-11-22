/* ufsm img-rankavg OUT.tif A.tif B.tif [C.tif ..] [--w wa,wb,..] (inksdk 2026-11-21, the lead's 3-D ensemble test: the voxel head's render and
   the surface head's map of one crop, different resolutions): every input resampled (bilinear) onto A's pixel grid, a pixel valid where every
   input is > 0 there (nearest: 0 = no data, a surface map's px with no patch, a render outside the volume); per input the mid-rank of its value
   among the valid pixels (256-bin histogram: ties share their mid-rank) in [0, 1]; the weighted mean of the ranks -> OUT uint8 1 + round(253 r),
   0 where invalid. Rank averaging puts heads with different calibration on one scale (AUC depends on the order only). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "imgops.h"
#include "optcheck.h"
#include "tiffw.h"

int cmd_img_rankavg(int argc, char **argv) {
    const char *in[16]; int n = 0; double wt[16]; const char *ws = nullptr;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], opt_n("--w")) && a + 1 < argc) { ws = argv[++a]; continue; }
        if (n >= 16) { fprintf(stderr, "img-rankavg: at most 15 inputs\n"); return 2; }
        in[n++] = argv[a];
    }
    if (n < 3) { fprintf(stderr, "usage: ufsm img-rankavg OUT.tif A.tif B.tif [..] [--w wa,wb,..]: per-image mid-ranks over the px valid in all (> 0),\n"
                                 "  resampled onto A's grid (bilinear), weighted mean -> uint8 1 + 253 r (0 = invalid)\n"); return 2; }
    const char *out = in[0]; const int ni = n - 1;
    for (int k = 0; k < ni; k++) wt[k] = 1;
    if (ws) { const char *p = ws; for (int k = 0; k < ni; k++) { char *e; wt[k] = strtod(p, &e); if (e == p || wt[k] < 0) { fprintf(stderr, "img-rankavg: --w needs %d weights >= 0\n", ni); return 2; } p = *e == ',' ? e + 1 : e; } }
    double wsum = 0; for (int k = 0; k < ni; k++) wsum += wt[k]; if (!(wsum > 0)) { fprintf(stderr, "img-rankavg: weights sum to 0\n"); return 2; }
    int W, H; uint8_t *im[16] = {0}; int iw[16], ih[16];
    for (int k = 0; k < ni; k++) { im[k] = img_read_u8(in[k + 1], &iw[k], &ih[k]); if (!im[k]) { fprintf(stderr, "img-rankavg: cannot read %s\n", in[k + 1]); return 1; } }
    W = iw[0]; H = ih[0]; const size_t np = (size_t)W * H;
    uint8_t *v = malloc(np * ni), *ok = malloc(np);   /* each input's value on A's grid, interleaved */
    for (size_t i = 0; i < np; i++) ok[i] = 1;
    for (int k = 0; k < ni; k++) {
        const double sx = (double)iw[k] / W, sy = (double)ih[k] / H;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = (size_t)y * W + x; uint8_t val;
            if (!k) val = im[0][i];
            else { double fx = (x + 0.5) * sx - 0.5, fy = (y + 0.5) * sy - 0.5; if (fx < 0) fx = 0; if (fy < 0) fy = 0; if (fx > iw[k] - 1) fx = iw[k] - 1; if (fy > ih[k] - 1) fy = ih[k] - 1;
                const int x0 = (int)fx, y0 = (int)fy, x1 = x0 + 1 < iw[k] ? x0 + 1 : x0, y1 = y0 + 1 < ih[k] ? y0 + 1 : y0; const double ax = fx - x0, ay = fy - y0;
                const uint8_t *m = im[k]; const size_t w = iw[k];
                const int nx = (int)lround(fx), ny = (int)lround(fy);
                if (!m[(size_t)ny * w + nx]) val = 0;   /* (validity by the nearest px: no blending of data into a no-data border) */
                else {   /* bilinear over the neighbours with data only (renormalised): a 0 neighbour would pull border px down and shift every rank */
                    const uint8_t q[4] = {m[y0 * w + x0], m[y0 * w + x1], m[y1 * w + x0], m[y1 * w + x1]};
                    const double g[4] = {(1 - ay) * (1 - ax), (1 - ay) * ax, ay * (1 - ax), ay * ax};
                    double s = 0, sw = 0; for (int j = 0; j < 4; j++) if (q[j]) { s += g[j] * q[j]; sw += g[j]; }
                    val = sw > 0 ? (uint8_t)fmax(1, fmin(255, lround(s / sw))) : m[(size_t)ny * w + nx]; } }
            v[i * ni + k] = val; if (!val) ok[i] = 0;
        }
    }
    double rk[16][256]; size_t nv = 0;
    for (int k = 0; k < ni; k++) { uint64_t h[256] = {0}; nv = 0;
        for (size_t i = 0; i < np; i++) if (ok[i]) { h[v[i * ni + k]]++; nv++; }
        uint64_t c = 0; for (int b = 0; b < 256; b++) { rk[k][b] = nv ? (c + 0.5 * h[b]) / (double)nv : 0; c += h[b]; } }
    uint8_t *o = calloc(np, 1);
    for (size_t i = 0; i < np; i++) if (ok[i]) { double r = 0; for (int k = 0; k < ni; k++) r += wt[k] * rk[k][v[i * ni + k]]; o[i] = (uint8_t)(1 + lround(253 * r / wsum)); }
    fprintf(stderr, "img-rankavg: %d inputs on %d x %d, %zu valid px (%.1f %%) -> %s\n", ni, W, H, nv, 100.0 * nv / np, out);
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp%d", out, (int)getpid());
    const int rc = tiffw_write(tmp, W, H, 1, o, 1) || rename(tmp, out); if (rc) { fprintf(stderr, "img-rankavg: cannot write %s\n", out); unlink(tmp); }
    for (int k = 0; k < ni; k++) free(im[k]); free(v); free(ok); free(o);
    return rc ? 1 : 0;
}
