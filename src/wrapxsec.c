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

/* ---- ufsm wrap-align <tifxyz> --ct ROOT KEY --level L [--every 8] [--r 3] [--block 512] [--cos 0.9] [--out PREFIX]
   How well a mesh follows the CT's lamination (lead 11-26, PHerc1203: the global spiral cuts across folded bundles; read first where it
   follows the sheets). Every --every-th grid vertex (both directions): the mesh normal (cross product of the grid's row / column central
   differences) against the sheet normal of the CT at level L, the dominant eigenvector (power iteration) of the gradient structure tensor
   sum g g^T over the (2r + 1)^3 level-L voxels around the vertex (central differences of the raw uint8). |cos| >= --cos counts as aligned.
   Vertices on air (tensor trace below 1 % of the block's median... simply: mean |g|^2 < 4) or outside the volume are skipped.
   The grid is cut into --block x --block vertex blocks (each block's CT read once: its vertices' bounding box at level L + r + 1):
   PREFIX.tsv (row0 col0 rows cols n_samples n_aligned mean_cos aligned_share) and PREFIX.pgm (one pixel per block: aligned share x 255,
   0 = no sample); stderr: the whole mesh's aligned share. Output pixels of ufsm render --scale 1 on this mesh = grid vertices / scale. */
static double xs_dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
int cmd_wrap_align(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wrap-align <tifxyz> --ct ROOT KEY --level L [--every 8] [--r 3] [--block 512] [--cos 0.9] [--out PREFIX]\n"); return 2; }
    const char *dir = argv[2], *root = nullptr, *key = nullptr;
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) { root = argv[i + 1]; key = argv[i + 2]; }
    const int L = atoi(xs_opt(argc, argv, "--level", "2")), ev = atoi(xs_opt(argc, argv, "--every", "8")), R = atoi(xs_opt(argc, argv, "--r", "3")), B = atoi(xs_opt(argc, argv, "--block", "512"));
    const double cth = atof(xs_opt(argc, argv, "--cos", "0.9")), f = ldexp(1.0, -L); const char *pre = xs_opt(argc, argv, "--out", nullptr);
    if (!root || !key || ev < 1 || R < 1 || B < ev) { fprintf(stderr, "wrap-align: --ct ROOT KEY required; --every >= 1, --r >= 1, --block >= --every\n"); return 2; }
    char p[2100]; int w[3], h[3]; float *X[3]; const char *nmx[3] = {"x", "y", "z"};
    for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s.tif", dir, nmx[c]); if (!(X[c] = img_read_f32(p, &w[c], &h[c]))) { fprintf(stderr, "wrap-align: cannot read %s\n", p); return 1; } }
    const int gw = w[0], gh = h[0];
    store *s = store_open(root); z3 *v = s ? z3_open(s, key, nullptr) : nullptr; if (!v) { fprintf(stderr, "wrap-align: cannot open %s %s\n", root, key); return 1; }
    const z3_meta *m = z3_meta_of(v);
    const int nbr = (gh + B - 1) / B, nbc = (gw + B - 1) / B; uint8_t *map = calloc((size_t)nbr * nbc, 1);
    FILE *tf = nullptr; if (pre) { snprintf(p, sizeof p, "%s.tsv", pre); tf = fopen(p, "w"); if (!tf) { fprintf(stderr, "wrap-align: cannot write %s\n", p); return 1; } fprintf(tf, "row0\tcol0\trows\tcols\tn\taligned\tmean_cos\tshare\n"); }
    size_t tn = 0, ta = 0;
    #define XV(c, r_, k_) X[c][(size_t)(r_) * gw + (k_)]
    for (int br = 0; br < nbr; br++) for (int bc = 0; bc < nbc; bc++) {
        const int r0 = br * B, c0 = bc * B, r1 = r0 + B < gh ? r0 + B : gh, c1 = c0 + B < gw ? c0 + B : gw;
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}; int any = 0;
        for (int r = r0; r < r1; r += ev) for (int k = c0; k < c1; k += ev) { if (XV(2, r, k) < 0) continue; any = 1;
            for (int c = 0; c < 3; c++) { const double q = XV(c, r, k) * f; if (q < lo[c]) lo[c] = q; if (q > hi[c]) hi[c] = q; } }
        if (!any) continue;
        int64_t o[3], n[3]; const int ax[3] = {2, 1, 0};   /* zyx box from xyz */
        for (int d = 0; d < 3; d++) { const int c = ax[d]; int64_t a = (int64_t)floor(lo[c]) - R - 1, b = (int64_t)ceil(hi[c]) + R + 2;
            if (a < 0) a = 0; if (b > m->shape[d]) b = m->shape[d]; o[d] = a; n[d] = b - a; }
        if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0 || n[0] * n[1] * n[2] > (int64_t)1 << 31) continue;
        uint8_t *vol = malloc((size_t)(n[0] * n[1] * n[2])); if (!vol || z3_read(v, o, n, vol, 8)) { free(vol); continue; }
        size_t bn = 0, ba = 0; double bsum = 0;
        for (int r = r0; r < r1; r += ev) for (int k = c0; k < c1; k += ev) {
            if (XV(2, r, k) < 0) continue;
            const int ra = r > 0 && XV(2, r - 1, k) >= 0 ? r - 1 : r, rb = r + 1 < gh && XV(2, r + 1, k) >= 0 ? r + 1 : r;
            const int ka = k > 0 && XV(2, r, k - 1) >= 0 ? k - 1 : k, kb = k + 1 < gw && XV(2, r, k + 1) >= 0 ? k + 1 : k;
            if (ra == rb || ka == kb) continue;
            double du[3], dv[3], nw[3];
            for (int c = 0; c < 3; c++) { du[c] = XV(c, r, kb) - XV(c, r, ka); dv[c] = XV(c, rb, k) - XV(c, ra, k); }
            nw[0] = du[1] * dv[2] - du[2] * dv[1]; nw[1] = du[2] * dv[0] - du[0] * dv[2]; nw[2] = du[0] * dv[1] - du[1] * dv[0];
            const double nl = sqrt(xs_dot(nw, nw)); if (!(nl > 0)) continue;
            for (int c = 0; c < 3; c++) nw[c] /= nl;
            const int64_t cz = (int64_t)lround(XV(2, r, k) * f) - o[0], cy = (int64_t)lround(XV(1, r, k) * f) - o[1], cx = (int64_t)lround(XV(0, r, k) * f) - o[2];
            if (cz - R - 1 < 0 || cy - R - 1 < 0 || cx - R - 1 < 0 || cz + R + 1 >= n[0] || cy + R + 1 >= n[1] || cx + R + 1 >= n[2]) continue;
            double T[3][3] = {{0}}; int cnt = 0;
            for (int64_t zz = cz - R; zz <= cz + R; zz++) for (int64_t yy = cy - R; yy <= cy + R; yy++) for (int64_t xx = cx - R; xx <= cx + R; xx++) {
                #define VV(a, b, c) (double)vol[((a) * n[1] + (b)) * n[2] + (c)]
                const double g[3] = {VV(zz, yy, xx + 1) - VV(zz, yy, xx - 1), VV(zz, yy + 1, xx) - VV(zz, yy - 1, xx), VV(zz + 1, yy, xx) - VV(zz - 1, yy, xx)};   /* x y z */
                #undef VV
                for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) T[a][b] += g[a] * g[b];
                cnt++;
            }
            if ((T[0][0] + T[1][1] + T[2][2]) / cnt < 16) continue;   /* air / flat: mean |grad|^2 < 16 (|g| < 4 grey levels over 2 voxels) */
            /* the power iteration starts from T's largest column (inside the dominant eigenvector's span; a start at the mesh normal
               itself stays there when T is rank-1 and orthogonal to it: a mesh across the sheets then read as aligned) */
            int jc = 0; double best = -1; for (int j = 0; j < 3; j++) { const double cn = T[0][j] * T[0][j] + T[1][j] * T[1][j] + T[2][j] * T[2][j]; if (cn > best) { best = cn; jc = j; } }
            double e[3] = {T[0][jc], T[1][jc], T[2][jc]}; { const double el = sqrt(xs_dot(e, e)); for (int a = 0; a < 3; a++) e[a] /= el; }
            for (int it = 0; it < 24; it++) { double q[3]; for (int a = 0; a < 3; a++) q[a] = T[a][0] * e[0] + T[a][1] * e[1] + T[a][2] * e[2];
                const double ql = sqrt(xs_dot(q, q)); if (!(ql > 0)) break; for (int a = 0; a < 3; a++) e[a] = q[a] / ql; }
            const double cs = fabs(xs_dot(e, nw));
            bn++; bsum += cs; if (cs >= cth) ba++;
        }
        free(vol);
        if (!bn) continue;
        tn += bn; ta += ba; map[(size_t)br * nbc + bc] = (uint8_t)(1 + lround(254.0 * ba / bn));
        if (tf) fprintf(tf, "%d\t%d\t%d\t%d\t%zu\t%zu\t%.4f\t%.4f\n", r0, c0, r1 - r0, c1 - c0, bn, ba, bsum / bn, (double)ba / bn);
    }
    #undef XV
    if (tf) fclose(tf);
    if (pre) { snprintf(p, sizeof p, "%s.pgm", pre); FILE *g = fopen(p, "wb"); if (g) { fprintf(g, "P5\n%d %d\n255\n", nbc, nbr); fwrite(map, 1, (size_t)nbr * nbc, g); fclose(g); } }
    fprintf(stderr, "wrap-align: %s: %zu samples, aligned (|cos| >= %.2f) %.4f, %d x %d blocks of %d vertices\n", dir, tn, cth, tn ? (double)ta / tn : 0.0, nbr, nbc, B);
    for (int c = 0; c < 3; c++) free(X[c]);
    free(map); z3_close(v); store_close(s);
    return 0;
}

