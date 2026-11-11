/* ufsm fiber-trace <root> <key-prefix> <out.fib> --box Z,Y,X,NZ,NY,NX [--scaledown 3] [--cache DIR] [--thr 64] [--bin 8]
       [--support 0.2] [--coh 0.65] [--turn 35] [--step 1] [--half 256] [--min-len 24] [--sep 1] [--threads 4]
   Streamlines through a published fibre prediction (open data <scroll>/representations/predictions/fibers/..: uint8 OME-zarr v2 arrays
   <prefix>_presence.ome.zarr/<s>, <prefix>_nx.ome.zarr/<s>, <prefix>_ny.ome.zarr/<s> at scaledown s, i.e. voxels of 2^s scan voxels).
   Encoding (verified against the uploader's Paris4_v1 streamlines, |cos| median 0.996 at presence >= 64, DESIGN.md 2026-11-10): the fibre
   axis is unoriented, stored on the z >= 0 hemisphere, t = (x, y, z) = ((nx - 128) / 127, (ny - 128) / 127, sqrt(1 - x^2 - y^2)).
   The tracer follows the community description of Paris4_v1 (seeds = the presence maximum of every --bin^3 cell at >= --thr, "bidirectional
   midpoint integration of unoriented dyadic fibre-axis tensors", support 0.2, coherence 0.65, 35 degrees a step, step one prediction
   voxel): the field at a point is the trilinear sum of presence-weighted dyads t t^T; its principal axis (power iteration from the
   incoming direction) is the direction, support = the interpolated presence / 255, coherence = lambda_1 / trace. A streamline stops when
   support or coherence fall below their thresholds, when a step turns more than --turn degrees, at the box or after --half steps each way.
   Seeds are taken by falling presence; a seed within --sep voxels (a cube) of an earlier streamline is skipped, so one fibre is traced
   once, not once per cell or once per voxel of its cross-section. Family: V when the mean squared axial component of the tangent is above 1/2, else H (the scroll axis is z).
   --presence-only 1 ignores nx / ny and takes the axis from the presence itself (the smallest eigenvector of its gradient structure
   tensor over 5^3 voxels); --mask ROOT KEY does the same on a zarr v3 probability volume (our own H / V mask predictions; <root> and
   <key-prefix> are then unused, pass -), with --family H|V naming the family of every line.
   Output: UFIB, points = level-0 scan voxels (z, y, x), voxel index = centre: (p + 0.5) 2^s - 0.5. --box is in prediction voxels.
   CPU; the reads are threaded, the tracing is one thread (the seed order makes it sequential). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "store.h"
#include "zarr2.h"
#include "zarr3.h"

static const char *ft_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
typedef struct { const uint8_t *pr, *nx, *ny; int64_t n[3]; } ft_vol;
/* the presence-weighted dyad field at p (box voxels): T (zz, yy, xx, zy, zx, yx) and the support; 0 outside */
static int ft_tensor(const ft_vol *V, const double p[3], double T[6], double *sup) {
    int64_t i0[3]; double f[3];
    for (int d = 0; d < 3; d++) { if (!(p[d] >= 0 && p[d] <= (double)(V->n[d] - 1))) return 0; i0[d] = (int64_t)p[d]; if (i0[d] > V->n[d] - 2) i0[d] = V->n[d] - 2; f[d] = p[d] - (double)i0[d]; }
    for (int k = 0; k < 6; k++) T[k] = 0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const double w = (c & 4 ? f[0] : 1 - f[0]) * (c & 2 ? f[1] : 1 - f[1]) * (c & 1 ? f[2] : 1 - f[2]); if (w == 0) continue;
        const size_t i = ((size_t)(i0[0] + (c >> 2)) * V->n[1] + (size_t)(i0[1] + ((c >> 1) & 1))) * V->n[2] + (size_t)(i0[2] + (c & 1));
        const double a = w * V->pr[i] / 255.0; if (a == 0) continue;
        const double x = (V->nx[i] - 128) / 127.0, y = (V->ny[i] - 128) / 127.0, zz = 1 - x * x - y * y, z = zz > 0 ? sqrt(zz) : 0;
        s += a; T[0] += a * z * z; T[1] += a * y * y; T[2] += a * x * x; T[3] += a * z * y; T[4] += a * z * x; T[5] += a * y * x;
    }
    *sup = s; return 1;
}
/* principal axis of T by power iteration from d0 (z, y, x); returns the coherence lambda_1 / trace */
static double ft_axis(const double T[6], const double d0[3], double d[3]) {
    double v[3] = {d0[0], d0[1], d0[2]}; const double tr = T[0] + T[1] + T[2]; if (!(tr > 0)) return 0;
    double lam = 0;
    for (int it = 0; it < 24; it++) {
        const double w[3] = {T[0] * v[0] + T[3] * v[1] + T[4] * v[2], T[3] * v[0] + T[1] * v[1] + T[5] * v[2], T[4] * v[0] + T[5] * v[1] + T[2] * v[2]};
        const double n = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(n > 1e-12 * tr)) return 0;
        const double e = fabs(n - lam); lam = n; for (int k = 0; k < 3; k++) v[k] = w[k] / n;
        if (e < 1e-6 * tr && it > 1) break;
    }
    for (int k = 0; k < 3; k++) d[k] = v[k];
    return lam / tr;
}
static int ft_flag(int argc, char **argv, const char *k) { for (int i = 5; i < argc; i++) if (!strcmp(argv[i], k)) return i; return 0; }
/* The fibre axis from the presence alone (--presence-only, --mask): the gradient structure tensor of the presence (central differences,
   products summed over a 5^3 box) has its SMALLEST eigenvalue along a tube; that eigenvector, on the z >= 0 hemisphere, is written
   in the published encoding (nx, ny). Streamed in z: five planes of the six in-plane-filtered products are kept. */
static void ft_ridge_axis(const uint8_t *pr, const int64_t n[3], uint8_t *nx, uint8_t *ny, int R, char prior) {
    const int64_t H = n[1], W = n[2]; const size_t hw = (size_t)H * W; const int RW = 2 * R + 1; float *ring = malloc((size_t)RW * 6 * hw * sizeof(float)), *tmp = malloc(hw * sizeof(float)), *raw = malloc(6 * hw * sizeof(float));
    #define PRV(z, y, x) ((float)pr[((size_t)(z < 0 ? 0 : z >= n[0] ? n[0] - 1 : z) * H + (size_t)(y < 0 ? 0 : y >= H ? H - 1 : y)) * W + (size_t)(x < 0 ? 0 : x >= W ? W - 1 : x)])
    for (int64_t zz = 0; zz < n[0] + R; zz++) {
        if (zz < n[0]) {   /* plane zz: products, then the separable (2R+1)^2 in-plane box sum, into its ring slot */
            float *Rg = ring + (size_t)(zz % RW) * 6 * hw;
            for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { const size_t i = (size_t)y * W + x;
                const float gz = PRV(zz + 1, y, x) - PRV(zz - 1, y, x), gy = PRV(zz, y + 1, x) - PRV(zz, y - 1, x), gx = PRV(zz, y, x + 1) - PRV(zz, y, x - 1);
                raw[i] = gz * gz; raw[hw + i] = gy * gy; raw[2 * hw + i] = gx * gx; raw[3 * hw + i] = gz * gy; raw[4 * hw + i] = gz * gx; raw[5 * hw + i] = gy * gx; }
            for (int c = 0; c < 6; c++) { const float *a = raw + c * hw; float *o = Rg + c * hw;
                for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { float s = 0; for (int64_t k = x - R; k <= x + R; k++) s += a[(size_t)y * W + (size_t)(k < 0 ? 0 : k >= W ? W - 1 : k)]; tmp[(size_t)y * W + x] = s; }
                for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { float s = 0; for (int64_t k = y - R; k <= y + R; k++) s += tmp[(size_t)(k < 0 ? 0 : k >= H ? H - 1 : k) * W + x]; o[(size_t)y * W + x] = s; } }
        }
        const int64_t z = zz - R; if (z < 0) continue;   /* planes z - R .. z + R are in the ring (clamped at the ends) */
        for (size_t i = 0; i < hw; i++) {
            double T[6] = {0, 0, 0, 0, 0, 0};
            for (int64_t k = z - R; k <= z + R; k++) { const int64_t kk = k < 0 ? 0 : k >= n[0] ? n[0] - 1 : k; const float *Rk = ring + (size_t)(kk % RW) * 6 * hw; for (int c = 0; c < 6; c++) T[c] += Rk[c * hw + i]; }
            const size_t j = (size_t)z * hw + i; nx[j] = ny[j] = 128; const double tr = T[0] + T[1] + T[2]; if (!(tr > 0)) continue;
            /* smallest eigenvector of T: the largest of (tr I - T), by power iteration (T is positive semidefinite) */
            const double M[6] = {tr - T[0], tr - T[1], tr - T[2], -T[3], -T[4], -T[5]}; double v[3] = {0.577, 0.577, 0.577};
            for (int it = 0; it < 40; it++) { const double w[3] = {M[0] * v[0] + M[3] * v[1] + M[4] * v[2], M[3] * v[0] + M[1] * v[1] + M[5] * v[2], M[4] * v[0] + M[5] * v[1] + M[2] * v[2]};
                const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; v[0] = w[0] / m; v[1] = w[1] / m; v[2] = w[2] / m; }
            if (prior) {   /* the layer's normal (largest eigenvector of T); the axis = the family's direction in that plane:
                              H: the horizontal in-plane direction normal x z; V: z minus its normal component (our predicted masks mark a
                              fibre LAYER, so the tube axis of the mask itself is ambiguous inside it; DESIGN.md 2026-11-11) */
                double u[3] = {0.577, 0.577, 0.577};
                for (int it = 0; it < 40; it++) { const double w[3] = {T[0] * u[0] + T[3] * u[1] + T[4] * u[2], T[3] * u[0] + T[1] * u[1] + T[5] * u[2], T[4] * u[0] + T[5] * u[1] + T[2] * u[2]};
                    const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; u[0] = w[0] / m; u[1] = w[1] / m; u[2] = w[2] / m; }
                double a[3];
                if (prior == 'H') { a[0] = 0; a[1] = u[2]; a[2] = -u[1]; }   /* u x (1, 0, 0) in (z, y, x): (0, u_x, -u_y) */
                else { a[0] = 1 - u[0] * u[0]; a[1] = -u[0] * u[1]; a[2] = -u[0] * u[2]; }
                const double m = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (m > 1e-6) for (int c = 0; c < 3; c++) v[c] = a[c] / m;
            }
            if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; }
            nx[j] = (uint8_t)lround(128 + 127 * v[2]); ny[j] = (uint8_t)lround(128 + 127 * v[1]);
        }
    }
    #undef PRV
    free(ring); free(tmp); free(raw);
}
/* --blur R (derived axes only): separable (2R+1)^3 box mean of the presence, in place (bridges short dips of a predicted tube) */
static void ft_blur(uint8_t *v, const int64_t n[3], int R) {
    const size_t nv = (size_t)n[0] * n[1] * n[2]; float *a = malloc(nv * sizeof(float)), *b = malloc(nv * sizeof(float)); if (!a || !b) { free(a); free(b); return; }
    for (size_t i = 0; i < nv; i++) a[i] = v[i];
    const size_t st[3] = {(size_t)n[1] * n[2], (size_t)n[2], 1};
    for (int d = 0; d < 3; d++) {
        for (size_t i = 0; i < nv; i++) { const int64_t c = (int64_t)(i / st[d] % (size_t)n[d]); double s = 0; int m = 0;
            for (int64_t k = c - R; k <= c + R; k++) if (k >= 0 && k < n[d]) { s += a[i + (size_t)(k - c) * st[d] * 1 - 0]; m++; }
            b[i] = (float)(s / m); }
        float *t = a; a = b; b = t;
    }
    for (size_t i = 0; i < nv; i++) v[i] = (uint8_t)lrintf(a[i] > 255 ? 255 : a[i]);
    free(a); free(b);
}
typedef struct { const ft_vol *V; double sup, coh, cturn, h; } ft_par;
/* the direction at p continuing prev (sign-aligned), 0 if the field ends there */
/* stop reasons (counted per streamline end): */
enum { FT_OK = 0, FT_SUP = 1, FT_COH = 2, FT_TURN = 3, FT_BOX = 4, FT_HALF = 5, FT_NSTOP = 6 };
static int ft_dir(const ft_par *P, const double p[3], const double prev[3], double d[3]) {
    double T[6], s; if (!ft_tensor(P->V, p, T, &s)) return FT_BOX; if (s < P->sup) return FT_SUP;
    if (ft_axis(T, prev, d) < P->coh) return FT_COH;
    const double c = d[0] * prev[0] + d[1] * prev[1] + d[2] * prev[2]; if (c < 0) for (int k = 0; k < 3; k++) d[k] = -d[k];
    return fabs(c) >= P->cturn ? FT_OK : FT_TURN;
}
/* a seed = (255 - presence) << 40 | voxel index: sorted ascending = falling presence, then position */
static int ft_seed_cmp(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

int cmd_fiber_trace(int argc, char **argv) {
    long long b[6];
    if (argc < 7 || sscanf(ft_opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        fprintf(stderr, "usage: ufsm fiber-trace <root> <key-prefix> <out.fib> --box Z,Y,X,NZ,NY,NX (prediction voxels) [--scaledown 3] [--cache DIR] [--thr 64] [--bin 8]\n"
                        "       [--support 0.2] [--coh 0.65] [--turn 35] [--step 1] [--half 256] [--min-len 24] [--sep 1] [--threads 4]\n");
        return 2;
    }
    const int sd = atoi(ft_opt(argc, argv, "--scaledown", "3")), thr = atoi(ft_opt(argc, argv, "--thr", "64")), bin = atoi(ft_opt(argc, argv, "--bin", "8")), half = atoi(ft_opt(argc, argv, "--half", "256"));
    const int nth = atoi(ft_opt(argc, argv, "--threads", "4")), sep = atoi(ft_opt(argc, argv, "--sep", "1")); const double minlen = atof(ft_opt(argc, argv, "--min-len", "24"));
    ft_par P = {.sup = atof(ft_opt(argc, argv, "--support", "0.2")), .coh = atof(ft_opt(argc, argv, "--coh", "0.65")), .cturn = cos(atof(ft_opt(argc, argv, "--turn", "35")) * M_PI / 180), .h = atof(ft_opt(argc, argv, "--step", "1"))};
    if (sd < 0 || sd > 8 || bin < 1 || half < 1 || sep < 0 || sep > 8 || !(P.h > 0)) { fprintf(stderr, "fiber-trace: bad options\n"); return 2; }
    const char *cache = ft_opt(argc, argv, "--cache", nullptr); uint8_t *vol[3] = {nullptr, nullptr, nullptr}; int64_t o[3], n[3];
    const int str = atoi(ft_opt(argc, argv, "--st-r", "2")), blur = atoi(ft_opt(argc, argv, "--blur", "0"));
    const int prior_on = atoi(ft_opt(argc, argv, "--axis-prior", "0"));
    const int bridge = atoi(ft_opt(argc, argv, "--bridge", "0")); size_t nbridge = 0;   /* steps straight on through a support dip */   /* with --family H|V: the axis from the layer normal + the family direction */
    if (str < 1 || str > 8 || blur < 0 || blur > 4) { fprintf(stderr, "fiber-trace: --st-r 1..8, --blur 0..4\n"); return 2; }
    const int mi = ft_flag(argc, argv, "--mask"), derive = mi || atoi(ft_opt(argc, argv, "--presence-only", "0")); const char famf = ft_opt(argc, argv, "--family", "?")[0];
    static const char *const nm[3] = {"presence", "nx", "ny"};
    if (mi) {   /* a mask probability (zarr v3 uint8, prob254: 255 unknown read as 0) as the presence */
        if (mi + 2 >= argc) { fprintf(stderr, "fiber-trace: --mask ROOT KEY\n"); return 2; }
        store *s = store_open(argv[mi + 1]); z3 *z = s ? z3_open(s, argv[mi + 2], cache) : nullptr; if (!z) { fprintf(stderr, "fiber-trace: cannot open the mask %s %s\n", argv[mi + 1], argv[mi + 2]); return 1; }
        /* --mask-origin Z,Y,X: the mask store's origin in --box coordinates (predict --box writes a box-local store; its origin_zyx) */
        long long mo[3] = {0, 0, 0}; const char *mos = ft_opt(argc, argv, "--mask-origin", nullptr);
        if (mos && sscanf(mos, "%lld,%lld,%lld", &mo[0], &mo[1], &mo[2]) != 3) { fprintf(stderr, "fiber-trace: --mask-origin Z,Y,X\n"); return 2; }
        int64_t lo[3];
        for (int d = 0; d < 3; d++) { const int64_t sh = z3_meta_of(z)->shape[d] + mo[d]; o[d] = b[d] < mo[d] ? mo[d] : b[d]; const int64_t e = b[d] + b[3 + d] > sh ? sh : b[d] + b[3 + d]; n[d] = e - o[d]; lo[d] = o[d] - mo[d]; }
        if (n[0] < 2 || n[1] < 2 || n[2] < 2) { fprintf(stderr, "fiber-trace: the box is outside the mask\n"); return 1; }
        vol[0] = malloc((size_t)n[0] * n[1] * n[2]);
        if (!vol[0] || z3_read(z, lo, n, vol[0], nth)) { fprintf(stderr, "fiber-trace: mask read failed\n"); return 1; }
        z3_close(z); for (size_t i = 0; i < (size_t)n[0] * n[1] * n[2]; i++) if (vol[0][i] == 255) vol[0][i] = 0;
    }
    for (int k = 0; k < 3 && !mi; k++) {
        if (k && derive) break;
        char key[1200]; snprintf(key, sizeof key, "%s_%s.ome.zarr/%d", argv[3], nm[k], sd);
        store *s = store_open(argv[2]); z2 *z = s ? z2_open(s, key, cache) : nullptr; if (!z) { fprintf(stderr, "fiber-trace: cannot open %s %s: %s\n", argv[2], key, z2_error()); return 1; }
        z2_assume_present(z, 1); const z2_meta *m = z2_meta_of(z);
        for (int d = 0; d < 3; d++) { o[d] = b[d] < 0 ? 0 : b[d]; const int64_t e = b[d] + b[3 + d] > m->shape[d] ? m->shape[d] : b[d] + b[3 + d]; n[d] = e - o[d]; }
        if (n[0] < 2 || n[1] < 2 || n[2] < 2) { fprintf(stderr, "fiber-trace: the box is outside the array (shape %lld %lld %lld)\n", (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2]); return 1; }
        vol[k] = malloc((size_t)n[0] * n[1] * n[2]);
        if (!vol[k] || z2_read(z, o, n, vol[k], nth)) { fprintf(stderr, "fiber-trace: read of %s failed: %s\n", key, z2_error()); return 1; }
        z2_close(z);
    }
    if (derive) { vol[1] = malloc((size_t)n[0] * n[1] * n[2]); vol[2] = malloc((size_t)n[0] * n[1] * n[2]); if (!vol[1] || !vol[2]) return 1; if (blur > 0) ft_blur(vol[0], n, blur);
        ft_ridge_axis(vol[0], n, vol[1], vol[2], str, prior_on && (famf == 'H' || famf == 'V') ? famf : 0); }
    const ft_vol V = {vol[0], vol[1], vol[2], {n[0], n[1], n[2]}}; P.V = &V; const size_t nv = (size_t)n[0] * n[1] * n[2];
    /* seeds: the presence maximum of each bin^3 cell */
    const int64_t g[3] = {(n[0] + bin - 1) / bin, (n[1] + bin - 1) / bin, (n[2] + bin - 1) / bin}; uint64_t *sd_ = malloc(((size_t)g[0] * g[1] * g[2] + 1) * sizeof *sd_); size_t ns = 0, npres = 0;
    for (int64_t cz = 0; cz < g[0]; cz++) for (int64_t cy = 0; cy < g[1]; cy++) for (int64_t cx = 0; cx < g[2]; cx++) {
        int best = thr - 1; size_t bi = 0;
        for (int64_t z = cz * bin; z < (cz + 1) * bin && z < n[0]; z++) for (int64_t y = cy * bin; y < (cy + 1) * bin && y < n[1]; y++) { const size_t r = ((size_t)z * n[1] + y) * n[2];
            for (int64_t x = cx * bin; x < (cx + 1) * bin && x < n[2]; x++) { const int v = V.pr[r + x]; npres += v >= thr; if (v > best) { best = v; bi = r + x; } } }
        if (best >= thr) sd_[ns++] = (uint64_t)(255 - best) << 40 | bi;
    }
    if (nv >= (1ull << 40)) { fprintf(stderr, "fiber-trace: the box has more than 2^40 voxels\n"); return 1; }
    qsort(sd_, ns, sizeof *sd_, ft_seed_cmp);
    uint8_t *occ = calloc((nv + 7) / 8, 1); fib_set F = {0}; size_t capr = 1 << 16, capp = 1 << 20;
    F.off = malloc((capr + 1) * sizeof(int)); F.fam = malloc(capr + 1); F.pts = malloc(capp * 12); F.off[0] = 0;
    double *line = malloc((size_t)(2 * half + 1) * 24); const double sc = ldexp(1.0, sd); size_t skipped = 0, shortl = 0, nstop[FT_NSTOP] = {0}; double tot = 0;
    for (size_t si = 0; si < ns; si++) {
        const size_t i = (size_t)(sd_[si] & ((1ull << 40) - 1)); if (occ[i >> 3] >> (i & 7) & 1) { skipped++; continue; }
        const double p0[3] = {(double)(i / ((size_t)n[1] * n[2])), (double)(i / n[2] % n[1]), (double)(i % n[2])};
        const double x = (V.nx[i] - 128) / 127.0, y = (V.ny[i] - 128) / 127.0, zz = 1 - x * x - y * y; const double d0[3] = {zz > 0 ? sqrt(zz) : 0, y, x};
        double ds[3]; { double T[6], s; if (!ft_tensor(&V, p0, T, &s) || s < P.sup || ft_axis(T, d0, ds) < P.coh) continue; }
        /* the line: backward half reversed, the seed, forward half */
        int nb = 0, nf = 0, ends[2] = {FT_HALF, FT_HALF}; double *fw = line + (size_t)half * 3, *seedp = fw;
        for (int side = 0; side < 2; side++) {
            double p[3] = {p0[0], p0[1], p0[2]}, d[3] = {side ? -ds[0] : ds[0], side ? -ds[1] : ds[1], side ? -ds[2] : ds[2]};
            int why = FT_HALF;
            for (int k = 0; k < half; k++) {
                double d1[3], d2[3], pm[3]; int r = ft_dir(&P, p, d, d1);
                if (r == FT_SUP && bridge > 0 && k + bridge < half) {   /* --bridge K: straight on through a support dip of up to K steps */
                    double pb[3] = {p[0], p[1], p[2]}; int kb = 0, ok = 0;
                    for (kb = 1; kb <= bridge; kb++) { for (int a = 0; a < 3; a++) pb[a] += P.h * d[a]; if (ft_dir(&P, pb, d, d1) == FT_OK) { ok = 1; break; } }
                    if (ok) { for (int b = 1; b <= kb; b++) { double *q = side ? seedp - 3 * (++nb) : seedp + 3 * (++nf); for (int a = 0; a < 3; a++) q[a] = p[a] + b * P.h * d[a]; }
                        for (int a = 0; a < 3; a++) p[a] = pb[a]; k += kb; nbridge++; r = FT_OK; }
                }
                if (r) { why = r; break; }
                for (int a = 0; a < 3; a++) pm[a] = p[a] + 0.5 * P.h * d1[a];
                if ((r = ft_dir(&P, pm, d1, d2))) { why = r; break; }
                if (d2[0] * d[0] + d2[1] * d[1] + d2[2] * d[2] < P.cturn) { why = FT_TURN; break; }
                for (int a = 0; a < 3; a++) { p[a] += P.h * d2[a]; d[a] = d2[a]; }
                if (!(p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] <= (double)(n[0] - 1) && p[1] <= (double)(n[1] - 1) && p[2] <= (double)(n[2] - 1))) { why = FT_BOX; break; }
                double *q = side ? seedp - 3 * (++nb) : seedp + 3 * (++nf); q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
            }
            ends[side] = why;
        }
        seedp[0] = p0[0]; seedp[1] = p0[1]; seedp[2] = p0[2];
        const int np = nb + nf + 1; const double *q0 = seedp - 3 * nb; occ[i >> 3] |= (uint8_t)(1 << (i & 7));
        if ((np - 1) * P.h < minlen) { shortl++; continue; }
        nstop[ends[0]]++; nstop[ends[1]]++;
        double tz2 = 0;
        for (int k = 0; k < np; k++) { const double *q = q0 + 3 * k; const int64_t c[3] = {llround(q[0]), llround(q[1]), llround(q[2])};
            for (int64_t z = c[0] - sep; z <= c[0] + sep; z++) for (int64_t y = c[1] - sep; y <= c[1] + sep; y++) for (int64_t x = c[2] - sep; x <= c[2] + sep; x++) {
                if (z < 0 || y < 0 || x < 0 || z >= n[0] || y >= n[1] || x >= n[2]) continue;
                const size_t j = ((size_t)z * n[1] + (size_t)y) * n[2] + (size_t)x; occ[j >> 3] |= (uint8_t)(1 << (j & 7)); }
            if (k) { const double dz = q[0] - q[-3]; tz2 += dz * dz / (P.h * P.h); } }
        if ((size_t)F.n + 1 > capr) { capr *= 2; F.off = realloc(F.off, (capr + 1) * sizeof(int)); F.fam = realloc(F.fam, capr + 1); }
        if ((size_t)F.npts + np > capp) { capp = (capp + np) * 2; F.pts = realloc(F.pts, capp * 12); }
        if ((size_t)F.npts + np > 0x7fffffff) { fprintf(stderr, "fiber-trace: more than 2^31 points; trace a smaller box\n"); return 1; }
        for (int k = 0; k < np; k++) for (int a = 0; a < 3; a++) F.pts[3 * (size_t)(F.npts + k) + a] = (float)((q0[3 * k + a] + (double)o[a] + 0.5) * sc - 0.5);
        F.fam[F.n] = famf == 'H' || famf == 'V' ? famf : tz2 / (np - 1) > 0.5 ? 'V' : 'H'; F.npts += np; F.off[++F.n] = F.npts; tot += (np - 1) * P.h;
    }
    size_t nH = 0; for (int r = 0; r < F.n; r++) nH += F.fam[r] == 'H';
    printf("fiber-trace: box %lld %lld %lld + %lld %lld %lld at scaledown %d: presence >= %d in %.4f of the voxels, %zu seeds (%zu on an earlier line, %zu shorter than %.0f)\n",
           (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], sd, thr, (double)npres / (double)nv, ns, skipped, shortl, minlen);
    printf("fiber-trace: %d streamlines (%zu H, %zu V), %d points, total length %.0f prediction voxels, mean %.1f -> %s\n", F.n, nH, (size_t)F.n - nH, F.npts, tot, F.n ? tot / F.n : 0, argv[4]);
    printf("fiber-trace: line ends (kept lines): support %zu, coherence %zu, turn %zu, box %zu, half %zu; %zu bridged dips\n", nstop[FT_SUP], nstop[FT_COH], nstop[FT_TURN], nstop[FT_BOX], nstop[FT_HALF], nbridge);
    if (fib_save(argv[4], &F)) { fprintf(stderr, "fiber-trace: cannot write %s\n", argv[4]); return 1; }
    free(line); free(occ); free(sd_); fib_free(&F); for (int k = 0; k < 3; k++) free(vol[k]);
    return 0;
}
