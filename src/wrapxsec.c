/* ufsm wrap-xsec <out.jpg> --ct ROOT KEY --level L --z Z [--crop y,x,h,w] [--max-px N] [--dot 1] R,G,B:TIFXYZ ...
   One axial CT slice (level L of the volume whose level-0 voxels the meshes are in; Z in level-0 voxels) with the cross-sections of tifxyz
   meshes drawn over it, one colour per mesh (2026-11-26, PHerc1203 hand test, lead: "follow one wrap of each fit a full turn on a clean
   slice: the right hand stays on one physical sheet across the theta seam, the wrong one jumps a sheet once per turn").
   Cross-section: every grid edge (row and column neighbours, both vertices valid, z >= 0) whose end points lie on both sides of the plane
   is cut at the plane by linear interpolation; the cut point is drawn as a (2 dot + 1)^2 square. CT: the slice's 1st / 99th percentiles of
   its non-zero voxels stretched to 0..255, grey. crop: level voxels (default the whole slice). Output: RGB JPEG q90, longest side <= max-px. */
#include "imgops.h"
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *xs_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static int xs_is_opt(const char *a) { return !strcmp(a, opt_n("--ct")) || !strcmp(a, opt_n("--level")) || !strcmp(a, opt_n("--z")) || !strcmp(a, opt_n("--crop")) || !strcmp(a, opt_n("--max-px")) || !strcmp(a, opt_n("--dot")); }

int cmd_wrap_xsec(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm wrap-xsec <out.jpg> --ct ROOT KEY --level L --z Z (level-0) [--crop y,x,h,w (level voxels)] [--max-px 4000] [--dot 1] R,G,B:TIFXYZ ...\n");
        return 2;
    }
    const char *out = argv[2], *root = nullptr, *key = nullptr;
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) { root = argv[i + 1]; key = argv[i + 2]; }
    const int L = atoi(xs_opt(argc, argv, "--level", "0")), maxpx = atoi(xs_opt(argc, argv, "--max-px", "4000")), dot = atoi(xs_opt(argc, argv, "--dot", "1"));
    const char *zs = xs_opt(argc, argv, "--z", nullptr), *cs = xs_opt(argc, argv, "--crop", nullptr);
    if (!root || !key || !zs || L < 0) { fprintf(stderr, "wrap-xsec: --ct ROOT KEY, --level and --z are required\n"); return 2; }
    const double Z0 = atof(zs), f = ldexp(1.0, -L);
    store *s = store_open(root); if (!s) { fprintf(stderr, "wrap-xsec: cannot open %s\n", root); return 1; }
    z3 *v = z3_open(s, key, nullptr); if (!v) { fprintf(stderr, "wrap-xsec: cannot open %s %s: %s\n", root, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(v);
    int64_t cy = 0, cx = 0, ch = m->shape[1], cw = m->shape[2];
    if (cs && sscanf(cs, "%ld,%ld,%ld,%ld", &cy, &cx, &ch, &cw) != 4) { fprintf(stderr, "wrap-xsec: --crop y,x,h,w\n"); return 2; }
    const int64_t zl = (int64_t)floor(Z0 * f + 0.5);
    if (zl < 0 || zl >= m->shape[0] || cy < 0 || cx < 0 || ch <= 0 || cw <= 0 || cy + ch > m->shape[1] || cx + cw > m->shape[2]) { fprintf(stderr, "wrap-xsec: z / crop outside the volume (%ld x %ld x %ld)\n", m->shape[0], m->shape[1], m->shape[2]); return 2; }
    const int W = (int)cw, H = (int)ch; uint8_t *g = malloc((size_t)W * H), *rgb = malloc((size_t)W * H * 3);
    const int64_t o[3] = {zl, cy, cx}, n[3] = {1, ch, cw};
    if (!g || !rgb || z3_read(v, o, n, g, 8)) { fprintf(stderr, "wrap-xsec: slice read failed: %s\n", z3_error()); return 1; }
    size_t hist[256] = {0}, nz = 0; for (size_t i = 0; i < (size_t)W * H; i++) if (g[i]) { hist[g[i]]++; nz++; }
    int lo = 1, hi = 255; size_t acc = 0;
    for (int k = 1; k < 256; k++) { acc += hist[k]; if (acc * 100 >= nz) { lo = k; break; } }
    acc = 0; for (int k = 1; k < 256; k++) { acc += hist[k]; if (acc * 100 >= nz * 99) { hi = k; break; } }
    if (hi <= lo) hi = lo + 1;
    for (size_t i = 0; i < (size_t)W * H; i++) { int t = (g[i] - lo) * 255 / (hi - lo); t = t < 0 ? 0 : t > 255 ? 255 : t; rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = (uint8_t)t; }
    int nm = 0;
    for (int a = 3; a < argc; a++) {
        if (xs_is_opt(argv[a])) { a += !strcmp(argv[a], opt_n("--ct")) ? 2 : 1; continue; }
        int col[3]; char dir[2048];
        if (sscanf(argv[a], "%d,%d,%d:%2047s", &col[0], &col[1], &col[2], dir) != 4) { fprintf(stderr, "wrap-xsec: mesh spec R,G,B:DIR, got %s\n", argv[a]); return 2; }
        char p[2100]; int w[3], h[3]; float *X[3]; const char *nmx[3] = {"x", "y", "z"};
        for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s.tif", dir, nmx[c]); if (!(X[c] = img_read_f32(p, &w[c], &h[c]))) { fprintf(stderr, "wrap-xsec: cannot read %s\n", p); return 1; } }
        if (w[1] != w[0] || w[2] != w[0] || h[1] != h[0] || h[2] != h[0]) { fprintf(stderr, "wrap-xsec: %s: x / y / z grids differ in size\n", dir); return 1; }
        const int gw = w[0], gh = h[0]; size_t cuts = 0;
        for (int r = 0; r < gh; r++) for (int c = 0; c < gw; c++) for (int e = 0; e < 2; e++) {
            const int r2 = r + e, c2 = c + !e; if (r2 >= gh || c2 >= gw) continue;
            const size_t i = (size_t)r * gw + c, j = (size_t)r2 * gw + c2;
            const float za = X[2][i], zb = X[2][j]; if (za < 0 || zb < 0 || za == zb) continue;
            if ((za - Z0) * (zb - Z0) > 0) continue;
            const double t = (Z0 - za) / (zb - za), x = (X[0][i] + t * (X[0][j] - X[0][i])) * f - cx, y = (X[1][i] + t * (X[1][j] - X[1][i])) * f - cy;
            const int px = (int)lround(x), py = (int)lround(y);
            for (int dy = -dot; dy <= dot; dy++) for (int dx = -dot; dx <= dot; dx++) {
                const int qx = px + dx, qy = py + dy; if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
                uint8_t *q = rgb + 3 * ((size_t)qy * W + qx); q[0] = (uint8_t)col[0]; q[1] = (uint8_t)col[1]; q[2] = (uint8_t)col[2];
            }
            cuts++;
        }
        fprintf(stderr, "wrap-xsec: %s: %zu edge cuts at z %.1f\n", dir, cuts, Z0);
        for (int c = 0; c < 3; c++) free(X[c]);
        nm++;
    }
    int ow = W, oh = H;
    if (img_fit_rgb(&rgb, &ow, &oh, maxpx, 8) || jpeg_write_rgb(out, ow, oh, rgb, 90)) { fprintf(stderr, "wrap-xsec: cannot write %s\n", out); return 1; }
    fprintf(stderr, "wrap-xsec: %d meshes on %s level %d slice %ld (CT stretch %d..%d) -> %s (%d x %d)\n", nm, key, L, zl, lo, hi, out, ow, oh);
    free(g); free(rgb); z3_close(v); store_close(s);
    return 0;
}
