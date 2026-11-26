/* ufsm ct-resample (2026-11-22, the 1447 resolution test): a CT level resampled trilinearly to another voxel size over given boxes, written
   as a one-level OME group (volcomp q 0) whose scale is the new size, so predict / render read it at level 0 unchanged (--um U --level 0)
   and the crop meshes move onto it with tifxyz-scale (div = U / source um). Only the shards under the boxes are written (the rest is absent,
   reads as 0). Up-sampling only: down-sampling would need an anti-alias prefilter (the pyramid's mean pooling does that).
   Geometry: voxel centres aligned, as the 2x mean-pooled pyramid levels are: output voxel o sits at source position (o + 0.5) f - 0.5,
   f = U / source um; edges clamp. Rounded to nearest (floor(v + 0.5)). Test: tests/test_ct_resample.py (numpy reference).
   2026-11-25 (user: fine-resolution ink nets on 8 / 9 um scans upsampled to their rung): --mode cubic = separable Catmull-Rom (a = -0.5)
   at the same positions, the 4 taps' indices clamped to the volume (edge replicated), x then y then z passes in double, clamped to 0..255;
   --axis IN OUT writes the scan's axis json (control points in the source's level-0 voxels) in the output's voxels ((p + 0.5) u0 / U - 0.5),
   so predict --axis keeps the radial input channels in frame. */
#include "optcheck.h"
#include "sources.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}

typedef struct { int64_t i0, i1; double w; } rs_tap;   /* source indices (relative to the read region) and the weight of i1 */
typedef struct { const uint8_t *src; int64_t sn[3]; uint8_t *dst; int SH; const rs_tap *t[3]; int64_t a[3], e[3], o[3]; int z0, z1; } rs_job;

static void *rs_run(void *p) {
    const rs_job *J = p; const size_t sy = (size_t)J->sn[2], sz = (size_t)J->sn[1] * J->sn[2];
    for (int64_t z = J->a[0] + J->z0; z < J->a[0] + J->z1; z++) {
        const rs_tap tz = J->t[0][z - J->a[0]];
        for (int64_t y = J->a[1]; y < J->e[1]; y++) {
            const rs_tap ty = J->t[1][y - J->a[1]];
            const uint8_t *r00 = J->src + tz.i0 * sz + ty.i0 * sy, *r01 = J->src + tz.i0 * sz + ty.i1 * sy,
                          *r10 = J->src + tz.i1 * sz + ty.i0 * sy, *r11 = J->src + tz.i1 * sz + ty.i1 * sy;
            uint8_t *d = J->dst + ((size_t)(z - J->o[0]) * J->SH + (size_t)(y - J->o[1])) * J->SH - J->o[2];
            for (int64_t x = J->a[2]; x < J->e[2]; x++) {
                const rs_tap tx = J->t[2][x - J->a[2]];
                const double c00 = r00[tx.i0] + tx.w * (r00[tx.i1] - r00[tx.i0]), c01 = r01[tx.i0] + tx.w * (r01[tx.i1] - r01[tx.i0]),
                             c10 = r10[tx.i0] + tx.w * (r10[tx.i1] - r10[tx.i0]), c11 = r11[tx.i0] + tx.w * (r11[tx.i1] - r11[tx.i0]);
                const double c0 = c00 + ty.w * (c01 - c00), c1 = c10 + ty.w * (c11 - c10), v = c0 + tz.w * (c1 - c0);
                d[x] = (uint8_t)floor(v + 0.5);
            }
        }
    }
    return nullptr;
}

/* cubic: per output index the 4 source indices (relative to the read region, clamped) and their weights */
typedef struct { int64_t i[4]; double w[4]; } cs_tap;
static void cs_weights(double t, double w[4]) {
    const double a = -0.5, t2 = t * t, t3 = t2 * t;
    w[0] = a * (t3 - 2 * t2 + t); w[1] = (a + 2) * t3 - (a + 3) * t2 + 1; w[2] = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t; w[3] = a * (t3 - t2);
}
typedef struct { const uint8_t *src; int64_t sn[3]; double *bx, *by; uint8_t *dst; int SH; const cs_tap *t[3]; int64_t a[3], e[3], o[3]; int pass, k0, k1; } cs_job;
static void *cs_run(void *p) {   /* pass 0: x over the source rows [k0, k1) (z * sn1 + y); 1: y over source planes z in [k0, k1); 2: z over output z [k0, k1) */
    const cs_job *J = p; const int64_t nx = J->e[2] - J->a[2], ny = J->e[1] - J->a[1];
    if (J->pass == 0) for (int64_t r = J->k0; r < J->k1; r++) { const uint8_t *s = J->src + r * J->sn[2]; double *d = J->bx + r * nx;
        for (int64_t x = 0; x < nx; x++) { const cs_tap *t = &J->t[2][x]; d[x] = t->w[0] * s[t->i[0]] + t->w[1] * s[t->i[1]] + t->w[2] * s[t->i[2]] + t->w[3] * s[t->i[3]]; } }
    else if (J->pass == 1) for (int64_t z = J->k0; z < J->k1; z++) for (int64_t y = 0; y < ny; y++) { const cs_tap *t = &J->t[1][y];
        const double *s = J->bx + z * J->sn[1] * nx; double *d = J->by + (z * ny + y) * nx;
        for (int64_t x = 0; x < nx; x++) d[x] = t->w[0] * s[t->i[0] * nx + x] + t->w[1] * s[t->i[1] * nx + x] + t->w[2] * s[t->i[2] * nx + x] + t->w[3] * s[t->i[3] * nx + x]; }
    else for (int64_t z = J->k0; z < J->k1; z++) { const cs_tap *t = &J->t[0][z];
        for (int64_t y = 0; y < ny; y++) { uint8_t *d = J->dst + ((size_t)(z + J->a[0] - J->o[0]) * J->SH + (size_t)(y + J->a[1] - J->o[1])) * J->SH + (J->a[2] - J->o[2]);
            for (int64_t x = 0; x < nx; x++) { const size_t q = (size_t)y * nx + x, pz = (size_t)ny * nx;
                const double v = t->w[0] * J->by[t->i[0] * pz + q] + t->w[1] * J->by[t->i[1] * pz + q] + t->w[2] * J->by[t->i[2] * pz + q] + t->w[3] * J->by[t->i[3] * pz + q];
                d[x] = (uint8_t)(v <= 0 ? 0 : v >= 255 ? 255 : floor(v + 0.5)); } } }
    return nullptr;
}

int cmd_ct_resample(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm ct-resample ROOT KEY OUT --um U0 --to U [--level L] --box z,y,x,nz,ny,nx [--box ..] [--shard 256] [--threads 6] [--cache DIR] [--mode linear|cubic] [--axis IN OUT]\n"
                        "  KEY's level L (voxel size U0 x 2^L; boxes in its voxels) resampled trilinearly to U um (up-sampling only) over the boxes,\n"
                        "  written as a one-level OME group OUT (level named by U, volcomp q 0): predict OUT's parent, key basename, --um U --level 0\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const double u0 = opt_atof("--um", opt(argc, argv, "--um", "0")), U = opt_atof("--to", opt(argc, argv, "--to", "0"));
    const int L = opt_atoi("--level", opt(argc, argv, "--level", "0")), SH = opt_atoi("--shard", opt(argc, argv, "--shard", "256"));
    const int nth = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 6);
    const double su = u0 * (double)(1 << L), f = U / su;
    const char *mstr = opt(argc, argv, "--mode", "linear"); const int cubic = !strcmp(mstr, "cubic");
    if (!cubic && strcmp(mstr, "linear")) { fprintf(stderr, "ct-resample: --mode linear | cubic\n"); return 2; }
    if (!(u0 > 0) || !(U > 0)) { fprintf(stderr, "ct-resample: --um and --to must be positive\n"); return 2; }
    if (f > 1.0001) { fprintf(stderr, "ct-resample: %g um -> %g um is down-sampling (no anti-alias prefilter here): use the pyramid's level\n", su, U); return 2; }
    if (SH < 128 || SH % 128) { fprintf(stderr, "ct-resample: --shard must be a multiple of 128\n"); return 2; }
    int nb = 0; int64_t (*bx)[6] = calloc((size_t)argc, sizeof *bx);
    for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n("--box"))) {
        long long v[6]; if (sscanf(argv[i + 1], "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6 || v[3] <= 0 || v[4] <= 0 || v[5] <= 0) {
            fprintf(stderr, "ct-resample: bad --box %s (z,y,x,nz,ny,nx)\n", argv[i + 1]); return 2; }
        for (int d = 0; d < 6; d++) bx[nb][d] = v[d];
        nb++;
    }
    if (!nb) { fprintf(stderr, "ct-resample: no --box\n"); return 2; }
    store_global_init();
    store *st = store_open(root); if (!st) { fprintf(stderr, "ct-resample: cannot open %s\n", root); return 1; }
    z3 *z = pyramid_open_level(st, key, L, u0, opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "ct-resample: cannot open level %d (%g um) of %s/%s: %s\n", L, su, root, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    if (m->esize != 1) { fprintf(stderr, "ct-resample: %s is not uint8\n", key); return 1; }
    int64_t osh[3]; for (int d = 0; d < 3; d++) osh[d] = (int64_t)ceil((double)m->shape[d] / f - 1e-9);
    /* output boxes: the source box's extent / f, outward */
    int64_t (*ob)[6] = calloc((size_t)nb, sizeof *ob);
    for (int b = 0; b < nb; b++) for (int d = 0; d < 3; d++) {
        int64_t lo = (int64_t)floor((double)bx[b][d] / f + 1e-9), hi = (int64_t)ceil((double)(bx[b][d] + bx[b][3 + d]) / f - 1e-9);
        if (lo < 0) lo = 0; if (hi > osh[d]) hi = osh[d];
        ob[b][d] = lo; ob[b][3 + d] = hi;
    }
    char ln[64], ad[4096], at[1024]; z3w_level_name(U, ln, sizeof ln); snprintf(ad, sizeof ad, "%s/%s", out, ln);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"ufsm ct-resample: %s level %d (%g um) %s to %g um, centres aligned, %d boxes only\"}}", key, L, su, cubic ? "Catmull-Rom cubic" : "trilinear", U, nb);
    mkdir(out, 0755);
    if (z3w_write_group(out, &U, 1, key, nullptr)) { fprintf(stderr, "ct-resample: %s: %s\n", out, z3w_error()); return 1; }
    z3w *w = z3w_create(ad, osh, SH, 0.f, 0, at); if (!w) { fprintf(stderr, "ct-resample: %s: %s\n", ad, z3w_error()); return 1; }
    uint8_t *dst = malloc((size_t)SH * SH * SH); rs_tap *tap[3]; for (int d = 0; d < 3; d++) tap[d] = malloc((size_t)SH * sizeof *tap[d]);
    pthread_t *T = malloc((size_t)nth * sizeof *T); rs_job *J = malloc((size_t)nth * sizeof *J); cs_job *CJ = malloc((size_t)nth * sizeof *CJ);
    int64_t gs[3]; for (int d = 0; d < 3; d++) gs[d] = (osh[d] + SH - 1) / SH;
    long nsh = 0; double vox = 0, rvox = 0;
    for (int64_t gz = 0; gz < gs[0]; gz++) for (int64_t gy = 0; gy < gs[1]; gy++) for (int64_t gx = 0; gx < gs[2]; gx++) {
        const int64_t o[3] = {gz * SH, gy * SH, gx * SH};
        int64_t a[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, e[3] = {INT64_MIN, INT64_MIN, INT64_MIN}; int any = 0;
        for (int b = 0; b < nb; b++) { int64_t ba[3], be[3]; int in = 1;
            for (int d = 0; d < 3; d++) { ba[d] = ob[b][d] > o[d] ? ob[b][d] : o[d]; be[d] = ob[b][3 + d] < o[d] + SH ? ob[b][3 + d] : o[d] + SH; in &= be[d] > ba[d]; }
            if (in) { any = 1; for (int d = 0; d < 3; d++) { if (ba[d] < a[d]) a[d] = ba[d]; if (be[d] > e[d]) e[d] = be[d]; } } }
        if (!any) continue;
        int64_t s0[3], sn[3];
        if (cubic) {   /* the source region under [a, e) with the 4 taps' reach, the clamped taps, then the three separable passes */
            cs_tap *ct[3];
            for (int d = 0; d < 3; d++) {
                const int64_t n = m->shape[d];
                int64_t lo = (int64_t)floor((a[d] + 0.5) * f - 0.5) - 1, hi = (int64_t)floor((e[d] - 0.5) * f - 0.5) + 3;
                if (lo < 0) lo = 0; if (lo > n - 1) lo = n - 1; if (hi > n) hi = n; if (hi <= lo) hi = lo + 1;
                s0[d] = lo; sn[d] = hi - lo; ct[d] = malloc((size_t)(e[d] - a[d]) * sizeof *ct[d]);
                for (int64_t i = a[d]; i < e[d]; i++) {
                    const double s = (i + 0.5) * f - 0.5, fl = floor(s); cs_tap *t = &ct[d][i - a[d]]; cs_weights(s - fl, t->w);
                    for (int k = 0; k < 4; k++) { int64_t j = (int64_t)fl - 1 + k; if (j < 0) j = 0; if (j > n - 1) j = n - 1; t->i[k] = j - lo; }
                }
            }
            for (int d = 0; d < 3; d++) for (int64_t i = 0; i < e[d] - a[d]; i++) for (int k = 0; k < 4; k++) if (ct[d][i].i[k] < 0 || ct[d][i].i[k] >= sn[d]) {
                fprintf(stderr, "ct-resample: internal: cubic tap outside the read region\n"); return 1; }
            uint8_t *src = malloc((size_t)sn[0] * sn[1] * sn[2]);
            if (!src || z3_read(z, s0, sn, src, nth)) { fprintf(stderr, "ct-resample: read failed: %s\n", z3_error()); return 1; }
            rvox += (double)sn[0] * sn[1] * sn[2];
            const int64_t nx = e[2] - a[2], ny = e[1] - a[1];
            double *b1 = malloc((size_t)sn[0] * sn[1] * nx * sizeof(double)), *b2 = malloc((size_t)sn[0] * ny * nx * sizeof(double));
            if (!b1 || !b2) { fprintf(stderr, "ct-resample: out of memory\n"); return 1; }
            memset(dst, 0, (size_t)SH * SH * SH);
            for (int pass = 0; pass < 3; pass++) {
                const int64_t N = pass == 0 ? sn[0] * sn[1] : pass == 1 ? sn[0] : e[0] - a[0]; const int k = N < nth ? (int)N : nth;
                for (int t = 0; t < k; t++) {
                    CJ[t] = (cs_job){src, {sn[0], sn[1], sn[2]}, b1, b2, dst, SH, {ct[0], ct[1], ct[2]}, {a[0], a[1], a[2]}, {e[0], e[1], e[2]}, {o[0], o[1], o[2]}, pass,
                                                (int)(N * t / k), (int)(N * (t + 1) / k)};
                    pthread_create(&T[t], nullptr, cs_run, &CJ[t]);
                }
                for (int t = 0; t < k; t++) pthread_join(T[t], nullptr);
            }
            free(src); free(b1); free(b2); for (int d = 0; d < 3; d++) free(ct[d]);
            if (z3w_write_shard(w, gz, gy, gx, dst, nth)) { fprintf(stderr, "ct-resample: write failed: %s\n", z3w_error()); return 1; }
            nsh++; vox += (double)(e[0] - a[0]) * (e[1] - a[1]) * (e[2] - a[2]);
            continue;
        }
        /* the source region under [a, e) and the per-axis taps into it */
        for (int d = 0; d < 3; d++) {
            const int64_t n = m->shape[d];
            int64_t lo = (int64_t)floor((a[d] + 0.5) * f - 0.5), hi = (int64_t)floor((e[d] - 0.5) * f - 0.5) + 2;
            if (lo < 0) lo = 0; if (lo > n - 1) lo = n - 1; if (hi > n) hi = n; if (hi <= lo) hi = lo + 1;
            s0[d] = lo; sn[d] = hi - lo;
            for (int64_t i = a[d]; i < e[d]; i++) {
                const double s = (i + 0.5) * f - 0.5; int64_t i0 = (int64_t)floor(s); double wt = s - (double)i0; int64_t i1 = i0 + 1;
                if (s <= 0) { i0 = i1 = 0; wt = 0; } else if (i0 >= n - 1) { i0 = i1 = n - 1; wt = 0; }
                tap[d][i - a[d]] = (rs_tap){i0 - lo, i1 - lo, wt};
            }
        }
        uint8_t *src = malloc((size_t)sn[0] * sn[1] * sn[2]);
        if (!src || z3_read(z, s0, sn, src, nth)) { fprintf(stderr, "ct-resample: read failed: %s\n", z3_error()); return 1; }
        rvox += (double)sn[0] * sn[1] * sn[2];
        memset(dst, 0, (size_t)SH * SH * SH);
        const int nz = (int)(e[0] - a[0]), k = nz < nth ? nz : nth;
        for (int t = 0; t < k; t++) {
            J[t] = (rs_job){src, {sn[0], sn[1], sn[2]}, dst, SH, {tap[0], tap[1], tap[2]}, {a[0], a[1], a[2]}, {e[0], e[1], e[2]}, {o[0], o[1], o[2]}, nz * t / k, nz * (t + 1) / k};
            pthread_create(&T[t], nullptr, rs_run, &J[t]);
        }
        for (int t = 0; t < k; t++) pthread_join(T[t], nullptr);
        free(src);
        if (z3w_write_shard(w, gz, gy, gx, dst, nth)) { fprintf(stderr, "ct-resample: write failed: %s\n", z3w_error()); return 1; }
        nsh++; vox += (double)(e[0] - a[0]) * (e[1] - a[1]) * (e[2] - a[2]);
    }
    if (z3w_close(w)) { fprintf(stderr, "ct-resample: close failed: %s\n", z3w_error()); return 1; }
    fprintf(stderr, "ct-resample: %s level %d (%g um, %lld x %lld x %lld) -> %s (%g um, %lld x %lld x %lld): %d boxes, %ld shards of %d, %.1f Mvox resampled from %.1f Mvox\n",
            key, L, su, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], ad, U, (long long)osh[0], (long long)osh[1], (long long)osh[2], nb, nsh, SH, vox / 1e6, rvox / 1e6);
    free(dst); for (int d = 0; d < 3; d++) free(tap[d]); free(T); free(J); free(CJ); free(bx); free(ob);
    for (int i = 5; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--axis"))) {   /* the axis in the output's voxels */
        axis ax; if (axis_load(&ax, argv[i + 1])) { fprintf(stderr, "ct-resample: cannot load axis %s\n", argv[i + 1]); return 1; }
        FILE *fo = fopen(argv[i + 2], "w"); if (!fo) { perror(argv[i + 2]); return 1; }
        const double k = u0 / U;
        fprintf(fo, "{\"source\":\"%s\",\"transform\":\"ct-resample: (p + 0.5) * %.9g - 0.5 (%g um level-0 voxels -> %g um)\",\"control_points\":[", argv[i + 1], k, u0, U);
        for (int j = 0; j < ax.n; j++) fprintf(fo, "%s{\"z\":%.4f,\"y\":%.4f,\"x\":%.4f}", j ? "," : "", (ax.z[j] + 0.5) * k - 0.5, (ax.y[j] + 0.5) * k - 0.5, (ax.x[j] + 0.5) * k - 0.5);
        fprintf(fo, "]}\n"); fclose(fo); fprintf(stderr, "ct-resample: axis %s -> %s (%d points x %.6g)\n", argv[i + 1], argv[i + 2], ax.n, k);
        free(ax.z); free(ax.y); free(ax.x);
    }
    z3_close(z); store_close(st);
    return 0;
}
