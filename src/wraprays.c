/* ufsm wrap-rays <out.tsv> --ct ROOT KEY --level L --z Z --meshes DIR [--map reg.json --cf CFDIR] [--center y,x] [--rays 8] [--win-um 1000]
                  [--um 2.403] [--prom 12] [--sep 3] [--theta0 0]
   Sheet count vs fit count along radial lines of one axial CT slice (2026-11-26, PHerc1203, lead: "2-3 CT laminae between consecutive
   wraps: does the fit undercount sheets? count CT laminae vs fitted wraps along ~8 radial lines, packed vs open, against count-field's
   predicted density"). The meshes (tifxyz, x y z in level-0 voxels of the CT volume) are the fitted wraps; the CT is level L of that volume.
   Per ray (angle theta0 + 360 k / rays degrees, image angle atan2(y, x) about the centre): the CT profile sampled every level voxel
   (bilinear), smoothed ([1 2 1] / 4 twice); the object span = first .. last sample above the slice's threshold (midpoint of the Otsu split's class means); the span cut into
   --win-um windows. Per window:
     laminae  local maxima above the Otsu threshold with prominence >= --prom (grey levels, against the lower of the minima within 4 sep on
              either side), at least --sep level voxels apart (the higher kept): bright papyrus bands
     wraps    crossings of the ray with the meshes' cross-sections at Z (marching squares over each grid cell: the iso-line z = Z)
     cf       the count-field's predicted turns over the window, |W(r1) - W(r0)| of the fit model in CFDIR (fit_load: model.json delta.f32
              axis.f32) at the window ends mapped to its scan by --map (rmap_f2c; without --map the meshes' scan is the model's)
     air      share of the window's samples below the Otsu threshold; class open (air >= 0.25) / packed (air <= 0.10) / mid
   Centre: --center y,x (level-0), else the count-field model's axis at this slice (iterated through the map), else the slice's centre.
   Out: TSV z theta_deg r0_um r1_um laminae wraps cf air class; stderr per ray and per class: laminae / wraps / cf sums and ratios. */
#include "fit.h"
#include "imgops.h"
#include "optcheck.h"
#include "reg.h"
#include "store.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *wr_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
typedef struct { float ax, ay, bx, by; } wr_seg;   /* a cross-section segment, level-0 x y */
typedef struct { wr_seg *s; size_t n, cap; } wr_segs;
static int wr_push(wr_segs *v, float ax, float ay, float bx, float by) {
    if (v->n == v->cap) { size_t c = v->cap ? 2 * v->cap : 4096; wr_seg *t = realloc(v->s, c * sizeof *t); if (!t) return -1; v->s = t; v->cap = c; }
    v->s[v->n++] = (wr_seg){ax, ay, bx, by}; return 0;
}
/* one tifxyz's iso-line z = Z: per grid cell the cuts of its 4 edges (both vertices valid), paired in order (2 or 4 cuts) */
static int wr_mesh(const char *dir, double Z, wr_segs *out) {
    char p[2100]; int w[3], h[3]; float *X[3] = {nullptr, nullptr, nullptr}; const char *nm[3] = {"x", "y", "z"};
    for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s.tif", dir, nm[c]); if (!(X[c] = img_read_f32(p, &w[c], &h[c]))) { for (int k = 0; k < c; k++) free(X[k]); return -1; } }
    int rc = 0;
    if (w[1] != w[0] || w[2] != w[0] || h[1] != h[0] || h[2] != h[0]) rc = -1;
    const int gw = w[0], gh = h[0];
    for (int r = 0; !rc && r + 1 < gh; r++) for (int c = 0; c + 1 < gw; c++) {
        const size_t q[4] = {(size_t)r * gw + c, (size_t)r * gw + c + 1, (size_t)(r + 1) * gw + c + 1, (size_t)(r + 1) * gw + c};
        int ok = 1; for (int k = 0; k < 4; k++) ok &= X[2][q[k]] >= 0;
        if (!ok) continue;
        float cx[4], cy[4]; int nc = 0;
        for (int k = 0; k < 4; k++) {
            const size_t i = q[k], j = q[(k + 1) & 3]; const double za = X[2][i], zb = X[2][j];
            if (za == zb || (za - Z) * (zb - Z) > 0 || (zb == Z)) continue;   /* zb == Z: counted by the next edge as its za */
            const double t = (Z - za) / (zb - za);
            cx[nc] = (float)(X[0][i] + t * (X[0][j] - X[0][i])); cy[nc] = (float)(X[1][i] + t * (X[1][j] - X[1][i])); nc++;
        }
        if (nc >= 2 && wr_push(out, cx[0], cy[0], cx[1], cy[1])) rc = -1;
        if (nc == 4 && wr_push(out, cx[2], cy[2], cx[3], cy[3])) rc = -1;
    }
    for (int c = 0; c < 3; c++) free(X[c]);
    return rc;
}
static double wr_otsu(const size_t hist[256]) {   /* the Otsu split's two class means, their midpoint (a split's own level sits on the
                                                     edge of the lower class: a smoothed narrow gap never reaches it) */
    double tot = 0, sum = 0; for (int k = 1; k < 256; k++) { tot += hist[k]; sum += (double)k * hist[k]; }
    double wb = 0, sb = 0, best = -1, t = 64;
    for (int k = 1; k < 255; k++) {
        wb += hist[k]; if (wb == 0) continue; const double wf = tot - wb; if (wf == 0) break;
        sb += (double)k * hist[k]; const double mb = sb / wb, mf = (sum - sb) / wf, v = wb * wf * (mb - mf) * (mb - mf);
        if (v > best) { best = v; t = 0.5 * (mb + mf); }
    }
    return t;
}
static double wr_axis(const float *a, int na, double z) {
    if (z <= 0) return a[0]; if (z >= na - 1) return a[na - 1];
    const int i = (int)z; const double u = z - i; return a[i] * (1 - u) + a[i + 1] * u;
}
/* the count-field's W at a level-0 point (x y z) of the meshes' scan */
static double wr_W(const fit_model *m, const rmap *map, double k, double x, double y, double z) {   /* k: level-0 -> model voxels without a map */
    double f[3] = {x, y, z}, c[3] = {x * k, y * k, z * k};
    if (map) rmap_f2c(map, f, c);
    return fit_W(m, c[2] - (double)m->origin[0], c[1] - (double)m->origin[1], c[0] - (double)m->origin[2]);
}
static int wr_cmp_d(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int cmd_wrap_rays(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm wrap-rays <out.tsv> --ct ROOT KEY --level L --z Z (level-0) [--meshes DIR] [--map reg.json --cf CFDIR] [--center y,x]\n"
                        "       [--rays 8] [--theta0 0] [--win-um 1000] [--um 2.403] [--prom 12] [--sep 3]\n");
        return 2;
    }
    const char *out = argv[2], *root = nullptr, *key = nullptr;
    for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) { root = argv[i + 1]; key = argv[i + 2]; }
    const int L = atoi(wr_opt(argc, argv, "--level", "1")), nray = atoi(wr_opt(argc, argv, "--rays", "8")), sep = atoi(wr_opt(argc, argv, "--sep", "3"));
    const double um0 = atof(wr_opt(argc, argv, "--um", "2.403")), winum = atof(wr_opt(argc, argv, "--win-um", "1000")), prom = atof(wr_opt(argc, argv, "--prom", "12")),
                 th0 = atof(wr_opt(argc, argv, "--theta0", "0"));
    const char *zs = wr_opt(argc, argv, "--z", nullptr), *md = wr_opt(argc, argv, "--meshes", nullptr), *mp = wr_opt(argc, argv, "--map", nullptr),
               *cfd = wr_opt(argc, argv, "--cf", nullptr), *ctr = wr_opt(argc, argv, "--center", nullptr);
    if (!root || !key || !zs || L < 0 || nray < 1 || sep < 1 || winum <= 0) { fprintf(stderr, "wrap-rays: --ct ROOT KEY, --z are required\n"); return 2; }
    const double Z = atof(zs), f = ldexp(1.0, -L), sum = um0 / f;   /* sum: um per level-L voxel */
    store *s = store_open(root); if (!s) { fprintf(stderr, "wrap-rays: cannot open %s\n", root); return 1; }
    z3 *v = z3_open(s, key, nullptr); if (!v) { fprintf(stderr, "wrap-rays: cannot open %s %s: %s\n", root, key, z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(v);
    const int64_t zl = (int64_t)floor(Z * f + 0.5), H = zm->shape[1], W = zm->shape[2];
    if (zl < 0 || zl >= zm->shape[0]) { fprintf(stderr, "wrap-rays: z outside the volume\n"); return 2; }
    uint8_t *g = malloc((size_t)H * W); const int64_t o[3] = {zl, 0, 0}, n[3] = {1, H, W};
    if (!g || z3_read(v, o, n, g, 8)) { fprintf(stderr, "wrap-rays: slice read failed: %s\n", z3_error()); return 1; }
    size_t hist[256] = {0}; for (size_t i = 0; i < (size_t)H * W; i++) if (g[i]) hist[g[i]]++;
    const double thr = wr_otsu(hist); const float at = (float)thr;
    /* the count-field model and the map */
    fit_model *cf = nullptr; const rmap *map = nullptr;
    if (cfd && !(cf = fit_load(cfd))) { fprintf(stderr, "wrap-rays: cannot load the fit model %s\n", cfd); return 1; }
    if (mp && !(map = reg_map_load(mp))) { fprintf(stderr, "wrap-rays: cannot load the map %s\n", mp); return 1; }
    const double ck = cf && !map && cf->um > 0 ? um0 / cf->um : 1;   /* same scan, model at its own voxel (1667 u1: 9.596 um = level 2) */
    /* centre (level-0 y x of the meshes' scan) */
    double cy0 = H / f / 2, cx0 = W / f / 2; const char *csrc = "slice centre";
    if (ctr) { if (sscanf(ctr, "%lf,%lf", &cy0, &cx0) != 2) { fprintf(stderr, "wrap-rays: --center y,x\n"); return 2; } csrc = "given"; }
    else if (cf) {
        const int na = (int)(cf->D * cf->h) + 1;
        for (int it = 0; it < 4; it++) {
            double fp[3] = {cx0, cy0, Z}, cp[3] = {cx0 * ck, cy0 * ck, Z * ck};
            if (map) rmap_f2c(map, fp, cp);
            const double bz = cp[2] - (double)cf->origin[0];
            double c2[3] = {wr_axis(cf->cx, na, bz) + (double)cf->origin[2], wr_axis(cf->cy, na, bz) + (double)cf->origin[1], cp[2]}, f2[3] = {c2[0] / ck, c2[1] / ck, c2[2] / ck};
            if (map) rmap_c2f(map, c2, f2);
            cx0 = f2[0]; cy0 = f2[1];
        }
        csrc = "count-field axis";
    }
    /* the meshes' cross-sections */
    wr_segs sg = {0}; int nmesh = 0;
    DIR *dd = md ? opendir(md) : nullptr; if (md && !dd) { fprintf(stderr, "wrap-rays: cannot open %s\n", md); return 1; }
    struct dirent *de;
    while (dd && (de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        char p[2100]; snprintf(p, sizeof p, "%s/%s", md, de->d_name);
        char q[2200]; snprintf(q, sizeof q, "%s/z.tif", p); FILE *t = fopen(q, "rb"); if (!t) continue; fclose(t);
        if (wr_mesh(p, Z, &sg)) { fprintf(stderr, "wrap-rays: cannot read the mesh %s\n", p); return 1; }
        nmesh++;
    }
    if (dd) closedir(dd);
    fprintf(stderr, "wrap-rays: slice %ld (level %d, %.2f um / voxel), threshold %.1f, centre y %.0f x %.0f (%s), %d meshes, %zu segments\n",
            (long)zl, L, sum, thr, cy0, cx0, csrc, nmesh, sg.n);
    FILE *fo = fopen(out, "w"); if (!fo) { fprintf(stderr, "wrap-rays: cannot write %s\n", out); return 1; }
    fprintf(fo, "z\ttheta_deg\tr0_um\tr1_um\tlaminae\tsheets\twraps\tcf\tair\tclass\n");
    double cls[3][4] = {{0}}; const char *cname[3] = {"packed", "mid", "open"};
    const int maxs = (int)(hypot((double)H, (double)W) + 2);
    float *pr = malloc(sizeof(float) * maxs), *sm = malloc(sizeof(float) * maxs); double *xr = malloc(sizeof(double) * (sg.n + 1)); uint8_t *pk = malloc(maxs);
    if (!pr || !sm || !xr || !pk) { fprintf(stderr, "wrap-rays: out of memory\n"); return 1; }
    for (int k = 0; k < nray; k++) {
        const double th = (th0 + 360.0 * k / nray) * M_PI / 180, dy = sin(th), dx = cos(th);
        const double py = cy0 * f, px = cx0 * f;   /* level-L centre */
        int ns = 0;
        for (; ns < maxs; ns++) {
            const double y = py + ns * dy, x = px + ns * dx; if (y < 0 || x < 0 || y > H - 1.001 || x > W - 1.001) break;
            const int64_t iy = (int64_t)y, ix = (int64_t)x; const double uy = y - iy, ux = x - ix; const uint8_t *b = g + iy * W + ix;
            pr[ns] = (float)((1 - uy) * ((1 - ux) * b[0] + ux * b[1]) + uy * ((1 - ux) * b[W] + ux * b[W + 1]));
        }
        if (ns < 8) continue;
        for (int pass = 0; pass < 2; pass++) {
            sm[0] = pr[0]; sm[ns - 1] = pr[ns - 1];
            for (int i = 1; i + 1 < ns; i++) sm[i] = 0.25f * pr[i - 1] + 0.5f * pr[i] + 0.25f * pr[i + 1];
            memcpy(pr, sm, sizeof(float) * ns);
        }
        int a0 = 0, a1 = ns - 1; while (a0 < ns && pr[a0] < at) a0++; while (a1 > a0 && pr[a1] < at) a1--;
        if (a1 - a0 < 8) continue;
        /* laminae: prominent maxima, >= sep apart (higher kept) */
        memset(pk, 0, ns); int last = -1000;
        for (int i = a0 + 1; i < a1; i++) {
            if (!(pr[i] >= pr[i - 1] && pr[i] > pr[i + 1] && pr[i] > at)) continue;
            float ml = pr[i], mr = pr[i];
            for (int j = i - 1; j >= a0 && j >= i - 4 * sep; j--) ml = fminf(ml, pr[j]);
            for (int j = i + 1; j <= a1 && j <= i + 4 * sep; j++) mr = fminf(mr, pr[j]);
            if (pr[i] - fmaxf(ml, mr) < prom) continue;
            if (i - last < sep) { if (pr[i] > pr[last]) { pk[last] = 0; pk[i] = 1; last = i; } continue; }
            pk[i] = 1; last = i;
        }
        /* air-separated laminae (pk 2): the valley to the previous lamina dips below the Otsu threshold (an air gap between two bands:
           separate sheets; two bands of one sheet, its recto / verso fibre layers, keep papyrus between them) */
        for (int i = a0, prev = -1; i <= a1; i++) {
            if (!pk[i]) continue;
            float mn = pr[i]; for (int j = prev < 0 ? i : prev; j < i; j++) mn = fminf(mn, pr[j]);
            if (prev < 0 || mn < at) pk[i] = 2;
            prev = i;
        }
        /* mesh crossings along the ray (level-0 radius from the centre) */
        size_t nx = 0;
        for (size_t j = 0; j < sg.n; j++) {
            const wr_seg *e = &sg.s[j]; const double ex = e->bx - e->ax, ey = e->by - e->ay, qx = e->ax - cx0, qy = e->ay - cy0;
            const double den = dx * ey - dy * ex; if (fabs(den) < 1e-12) continue;
            const double t = (qx * ey - qy * ex) / den, u = (qx * dy - qy * dx) / den;   /* ray point t (level 0), segment u in [0, 1) */
            if (t > 0 && u >= 0 && u < 1) xr[nx++] = t;
        }
        qsort(xr, nx, sizeof *xr, wr_cmp_d);
        const int wn = (int)fmax(1, floor(winum / sum + 0.5));
        double rl = 0, rsh = 0, rw = 0, rc = 0;
        for (int w0 = a0; w0 < a1; w0 += wn) {
            const int w1 = w0 + wn < a1 ? w0 + wn : a1; if (w1 - w0 < wn / 2) break;
            int lam = 0, shn = 0, air = 0; for (int i = w0; i < w1; i++) { lam += pk[i] > 0; shn += pk[i] == 2; air += pr[i] < at; }
            const double r0 = w0 / f, r1 = w1 / f; int wr = 0; for (size_t j = 0; j < nx; j++) wr += xr[j] >= r0 && xr[j] < r1;
            double cft = NAN;
            if (cf) cft = fabs(wr_W(cf, map, ck, cx0 + r1 * dx, cy0 + r1 * dy, Z) - wr_W(cf, map, ck, cx0 + r0 * dx, cy0 + r0 * dy, Z));
            const double as = (double)air / (w1 - w0); const int c = as <= 0.10 ? 0 : as >= 0.25 ? 2 : 1;
            fprintf(fo, "%.0f\t%.1f\t%.0f\t%.0f\t%d\t%d\t%d\t%.2f\t%.3f\t%s\n", Z, th * 180 / M_PI, r0 * um0, r1 * um0, lam, shn, wr, cft, as, cname[c]);
            rl += lam; rsh += shn; rw += wr; rc += isnan(cft) ? 0 : cft; cls[c][0] += lam; cls[c][1] += wr; cls[c][2] += isnan(cft) ? 0 : cft; cls[c][3] += shn;
        }
        fprintf(stderr, "wrap-rays: theta %6.1f: laminae %4.0f sheets %4.0f wraps %4.0f cf %6.1f  laminae / wraps %.2f  sheets / wraps %.2f  laminae / cf %.2f\n",
                th * 180 / M_PI, rl, rsh, rw, rc, rw > 0 ? rl / rw : NAN, rw > 0 ? rsh / rw : NAN, rc > 0 ? rl / rc : NAN);
    }
    for (int c = 0; c < 3; c++)
        fprintf(stderr, "wrap-rays: %-6s laminae %5.0f sheets %5.0f wraps %5.0f cf %7.1f  laminae / wraps %.2f  sheets / wraps %.2f  laminae / cf %.2f  wraps / cf %.2f\n", cname[c], cls[c][0], cls[c][3], cls[c][1], cls[c][2],
                cls[c][1] > 0 ? cls[c][0] / cls[c][1] : NAN, cls[c][1] > 0 ? cls[c][3] / cls[c][1] : NAN, cls[c][2] > 0 ? cls[c][0] / cls[c][2] : NAN, cls[c][2] > 0 ? cls[c][1] / cls[c][2] : NAN);
    fclose(fo); free(pr); free(sm); free(xr); free(pk); free(sg.s); free(g); z3_close(v); store_close(s);
    if (cf) fit_free(cf);
    return 0;
}
