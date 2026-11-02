/* ufsm recto-sp IN-FIELDS OUT-FIELDS --src ROOT KEY UM [--rows R0,R1] [--stride 4] [--half-um 400] [--prom 25] [--sigma 0.5] [--sep-um 20]
                [--blend 1] [--gate 0.4,2.5] [--min-net-um 0] [--min-int 3] [--rsp-in DIR] [--threads N] [--cache DIR]
   Recto predictions as a count target for count-field: the sheet spacing sp of a fit-field set replaced by the spacing of recto hits
   along the field normal. Per field cell on a --stride grid (cells with material): the source profile along the principal axis of the
   cell's n n^T over +-half-um (trilinear, step half a source voxel), peaks as recto-ruler (rr_peaks: smoothing --sigma source voxels,
   prominence --prom 0..255, merged closer than --sep-um), the median normal width of the up to 5 hit intervals around the cell centre
   (>= --min-int intervals) = one turn each (--cls MODEL: ufsm recto-cls turns per interval, spacing = their widths / turns) -> rsp (units of h / 8 native voxels, as sp; 0 none), written to the stride block.
   OUT gets sp = net^(1 - a) rsp^a (a = --blend) where rsp / net lies inside --gate, net sp elsewhere; rsp (the raw recto spacing); the
   other planes linked to IN's; fields.json copied. --rsp-in DIR: reuse DIR/rsp (only the blend runs). --rows R0,R1: box rows computed (the
   others keep the net sp). The source is on the fields' scan (UM its voxel um; native = fields' um). Arrays: 512^3 shards, lossless. */
#include "fit.h"
#include "rectoruler.h"
#include "sysinfo.h"
#include "zarr3.h"
#include "z3w.h"
#include "store.h"
#include "json.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>

static const char *rs_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double rs_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int rs_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

static int rs_axis(const uint8_t *const t[6], size_t i, double v[3]) {   /* principal eigenvector of n n^T (z y x) */
    double T[3][3]; const double a[6] = {t[0][i] / 255.0, t[1][i] / 255.0, t[2][i] / 255.0, t[3][i] / 255.0 - 0.5, t[4][i] / 255.0 - 0.5, t[5][i] / 255.0 - 0.5};
    T[0][0] = a[0]; T[1][1] = a[1]; T[2][2] = a[2]; T[0][1] = T[1][0] = a[3]; T[0][2] = T[2][0] = a[4]; T[1][2] = T[2][1] = a[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    for (int it = 0; it < 12; it++) {
        double w[3] = {0, 0, 0}; for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++) w[p] += T[p][q] * v[q];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-9) return -1; for (int p = 0; p < 3; p++) v[p] = w[p] / l;
    }
    return 0;
}

typedef struct {
    const fit_fields_buf *b; int r0, nr;   /* field rows [r0, r0 + nr) of the loaded block (box-local rows of b) */
    const uint8_t *src; int64_t so[3], sn[3]; double F;   /* source block (source voxels, global), F = native / source voxel */
    long long org[3]; int h, S; double um, half, prom, sigma, sep; int minint;
    uint8_t *out; int64_t oz0; int H, W;   /* output rows (box rows from oz0), H x W cells */
    const rr_cls *cls;   /* --cls: turns per hit interval from the classifier (features as recto-ruler's rr_feat, along the normal) */
    _Atomic long long ncell, nest;
} rs_job;

static inline double rs_tri(const rs_job *J, double z, double y, double x) {   /* source voxel coords (global) */
    z -= J->so[0]; y -= J->so[1]; x -= J->so[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= J->sn[0] || y0 + 1 >= J->sn[1] || x0 + 1 >= J->sn[2]) return NAN;
    const double tz = z - z0, ty = y - y0, tx = x - x0; const size_t sy = J->sn[2], sz = (size_t)J->sn[1] * J->sn[2];
    const uint8_t *q = J->src + z0 * sz + y0 * sy + x0;
    const double a = (1 - ty) * ((1 - tx) * q[0] + tx * q[1]) + ty * ((1 - tx) * q[sy] + tx * q[sy + 1]);
    const double c = (1 - ty) * ((1 - tx) * q[sz] + tx * q[sz + 1]) + ty * ((1 - tx) * q[sz + sy] + tx * q[sz + sy + 1]);
    return (1 - tz) * a + tz * c;
}

static void rs_rows(void *ctx, int a0, int a1, int t) {
    rs_job *J = ctx; (void)t;
    const fit_fields *f = &J->b->f; const int H = J->H, W = J->W, S = J->S;
    const double st = 0.5 / J->F;   /* native step = half a source voxel */
    const int n = 2 * (int)ceil(J->half / (st * J->um)) + 1, c = n / 2;
    float *v = malloc(sizeof(float) * n), *tmp = malloc(sizeof(float) * n), *spn = malloc(sizeof(float) * n), *ph = malloc(sizeof(float) * n); uint8_t *mat = malloc(n); memset(mat, 1, n); rr_hit *hp = malloc(sizeof(rr_hit) * (n / 2 + 2));
    long long nc = 0, ne = 0;
    for (int job = a0; job < a1; job++) {   /* job = (row block, y block) */
        const int nyb = (H + S - 1) / S, zb = job / nyb, yb = job % nyb; const int z = J->r0 + zb * S + S / 2, y = yb * S + S / 2;
        if (z >= J->r0 + J->nr) continue;
        const int zz = z < J->b->shape[0] ? z : J->b->shape[0] - 1, yy = y < H ? y : H - 1;
        for (int xb = 0; xb * S < W; xb++) {
            const int xx = xb * S + S / 2 < W ? xb * S + S / 2 : W - 1; const size_t ci = ((size_t)zz * H + yy) * W + xx;
            if (!f->w[ci]) continue;
            double nv[3]; if (rs_axis(f->t, ci, nv)) continue;
            nc++;
            /* the cell centre, native global (z y x) -> source voxels */
            const double p[3] = {J->org[0] + (zz + 0.5) * J->h, J->org[1] + (yy + 0.5) * J->h, J->org[2] + (xx + 0.5) * J->h};
            int ok = 1;
            for (int i = 0; i < n && ok; i++) { const double s = (i - c) * st; v[i] = (float)rs_tri(J, (p[0] + s * nv[0] + 0.5) * J->F - 0.5, (p[1] + s * nv[1] + 0.5) * J->F - 0.5, (p[2] + s * nv[2] + 0.5) * J->F - 0.5); }
            const int nh = rr_peaks(v, mat, n, J->sigma * 2, J->prom, (int)lround(60 / (st * J->um)), J->sep / (st * J->um), hp, tmp);
            int j = -1; for (int k = 0; k + 1 < nh; k++) if (hp[k].pos <= c && hp[k + 1].pos > c) { j = k; break; }
            if (j < 0) continue;
            float wd[5]; int nw = 0; for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh) wd[nw++] = hp[k + 1].pos - hp[k].pos;
            if (nw < J->minint) continue;
            double sp;
            if (J->cls) {   /* the intervals' turns from the classifier: spacing = sum of widths / sum of turns */
                for (int i = 0; i < n; i++) {   /* the fields along the profile (nearest cell): spacing (native), phase */
                    const double s = (i - c) * st; const int cz = (int)floor((p[0] + s * nv[0] - J->org[0]) / J->h), cy = (int)floor((p[1] + s * nv[1] - J->org[1]) / J->h), cx = (int)floor((p[2] + s * nv[2] - J->org[2]) / J->h);
                    spn[i] = 0; ph[i] = NAN;
                    if (cz < 0 || cy < 0 || cx < 0 || cz >= J->b->shape[0] || cy >= H || cx >= W) continue;
                    const size_t q = ((size_t)cz * H + cy) * W + cx; if (!f->w[q]) continue;
                    spn[i] = f->sp[q] * J->h / 8.0f; const int pa = f->pc[q] - 128, pb = f->ps[q] - 128; if (pa * pa + pb * pb >= 256) ph[i] = (float)(atan2(pb, pa) / (2 * M_PI));
                }
                double sw = 0, se = 0; int nint = 0;
                for (int k = j - 2; k <= j + 2; k++) {
                    if (k < 0 || k + 1 >= nh) continue;
                    float fe[RR_NF]; for (int e = 0; e < RR_NF; e++) fe[e] = NAN;
                    const double a = hp[k].pos, b = hp[k + 1].pos; const int ia = (int)ceil(a), ib = (int)floor(b), im = (int)lround(0.5 * (a + b));
                    const double wu = (b - a) * st * J->um; fe[0] = (float)wu; if (spn[im] > 0) { fe[1] = (float)(wu / (spn[im] * J->um)); fe[20] = (float)(spn[im] * J->um); }
                    { float nb[9]; int nn = 0; for (int q = k - 4; q <= k + 4; q++) if (q >= 0 && q + 1 < nh && q != k) nb[nn++] = (hp[q + 1].pos - hp[q].pos) * st * J->um;
                      if (nn >= 2) { qsort(nb, nn, sizeof(float), rs_cmpf); fe[2] = (float)(wu / nb[nn / 2]); } }
                    { double sdn = 0, sph = 0, vmin = 1e9, vs = 0; int ndn = 0, nph = 0, lp = -1, nv2 = 0, cnt = 0;
                      for (int i = ia; i <= ib; i++) { cnt++; if (spn[i] > 0) { sdn += st / spn[i]; ndn++; }
                          if (ph[i] == ph[i]) { if (lp >= 0 && i - lp <= 3) { double d = ph[i] - ph[lp]; d -= floor(d + 0.5); sph += d; nph += i - lp; } lp = i; }
                          if (v[i] == v[i]) { if (v[i] < vmin) vmin = v[i]; vs += v[i]; nv2++; } }
                      if (cnt > 0 && ndn >= 0.8 * cnt) fe[3] = (float)(sdn * (b - a) / ndn);
                      if (cnt > 1 && nph >= 0.8 * (cnt - 1)) fe[4] = (float)(sph * (b - a) / nph);
                      if (nv2) { fe[5] = (float)(vmin / 255); fe[6] = (float)(vs / nv2 / 255); } }
                    fe[7] = hp[k].h / 255; fe[8] = hp[k + 1].h / 255; fe[9] = hp[k].prom / 255; fe[10] = hp[k + 1].prom / 255;
                    fe[11] = (float)(hp[k].wid * st * J->um); fe[12] = (float)(hp[k + 1].wid * st * J->um);
                    if (k > 0) { const int qm = (int)lround(0.5 * (hp[k - 1].pos + hp[k].pos)); if (spn[qm] > 0) fe[17] = (float)((hp[k].pos - hp[k - 1].pos) * st / spn[qm]); }
                    if (k + 2 < nh) { const int qm = (int)lround(0.5 * (hp[k + 1].pos + hp[k + 2].pos)); if (spn[qm] > 0) fe[18] = (float)((hp[k + 2].pos - hp[k + 1].pos) * st / spn[qm]); }
                    fe[21] = 1;
                    float pr[RR_K]; rr_cls_predict(J->cls, fe, pr); double E = 0; for (int e = 0; e < RR_K; e++) E += e * pr[e];
                    sw += b - a; se += E; nint++;
                }
                if (nint < J->minint || se < 0.5) continue;
                sp = sw / se * st * 8 / J->h;
            } else {
                qsort(wd, nw, sizeof(float), rs_cmpf);
                sp = wd[nw / 2] * st * 8 / J->h;   /* (native voxels -> units of h / 8) */
            }
            const uint8_t q = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : lround(sp));
            ne++;
            for (int a = zb * S; a < zb * S + S && a < J->nr; a++) for (int b = yb * S; b < yb * S + S && b < H; b++) memset(J->out + ((size_t)(a + J->r0 - J->oz0) * H + b) * W + xb * S, q, (size_t)(xb * S + S <= W ? S : W - xb * S));
        }
    }
    atomic_fetch_add(&J->ncell, nc); atomic_fetch_add(&J->nest, ne);
    free(v); free(tmp); free(mat); free(hp); free(spn); free(ph);
}

static int rs_link(const char *in, const char *out, const char *name) {
    char a[2048], b[2048], r[4096]; snprintf(a, sizeof a, "%s/%s", in, name); snprintf(b, sizeof b, "%s/%s", out, name);
    if (access(a, F_OK)) return 0;
    if (!realpath(a, r)) return -1;
    unlink(b); return symlink(r, b);
}

int cmd_recto_sp(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm recto-sp IN-FIELDS OUT-FIELDS --src ROOT KEY UM [--rows R0,R1] [--stride 4] [--half-um 400] [--prom 25] [--sigma 0.5] [--sep-um 20]\n"
                                    "       [--blend 1] [--gate 0.4,2.5] [--min-int 3] [--rsp-in DIR] [--threads N] [--cache DIR]\n"); return 2; }
    const char *in = argv[2], *out = argv[3]; const int nt = ufsm_threads(atoi(rs_opt(argc, argv, "--threads", "0")), 16);
    fit_set_threads(nt);
    char path[4096]; snprintf(path, sizeof path, "%s/fields.json", in);
    FILE *fj = fopen(path, "rb"); if (!fj) { fprintf(stderr, "recto-sp: cannot read %s\n", path); return 1; }
    fseek(fj, 0, SEEK_END); long L = ftell(fj); fseek(fj, 0, SEEK_SET); char *jt = malloc(L + 1); if (fread(jt, 1, L, fj) != (size_t)L) { fclose(fj); return 1; } jt[L] = 0; fclose(fj);
    json *j = json_parse(jt, L); if (!j || strcmp(json_str(json_get(j, "format"), ""), "zarr")) { fprintf(stderr, "recto-sp: zarr fields only\n"); return 1; }
    long long org[3], zoff[3]; int shp[3]; for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); zoff[d] = (long long)json_num(json_at(json_get(j, "zarr_offset"), d), 0); }
    const int h = (int)json_num(json_get(j, "h"), 2); const double um = json_num(json_get(j, "um"), 9.6); json_free(j);
    mkdir(out, 0775);
    snprintf(path, sizeof path, "%s/fields.json", out); { FILE *o = fopen(path, "wb"); if (!o || fwrite(jt, 1, L, o) != (size_t)L) { fprintf(stderr, "recto-sp: cannot write %s\n", path); return 1; } fclose(o); } free(jt);
    const char *pl[] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "w", "dz", "dy", "dx", "dq", "r", "ct", "cs"};
    for (size_t k = 0; k < sizeof pl / sizeof *pl; k++) if (rs_link(in, out, pl[k])) { fprintf(stderr, "recto-sp: cannot link %s\n", pl[k]); return 1; }
    z3 *zsp = z3_open(store_open(in), "sp", nullptr); if (!zsp) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; }
    const int64_t *ash = z3_meta_of(zsp)->shape;
    int R0 = 0, R1 = shp[0]; { const char *r = rs_opt(argc, argv, "--rows", nullptr); if (r && sscanf(r, "%d,%d", &R0, &R1) != 2) { fprintf(stderr, "recto-sp: --rows R0,R1\n"); return 2; } if (R0 < 0) R0 = 0; if (R1 > shp[0]) R1 = shp[0]; }
    double blend = atof(rs_opt(argc, argv, "--blend", "1")), glo = 0.4, ghi = 2.5; sscanf(rs_opt(argc, argv, "--gate", "0.4,2.5"), "%lf,%lf", &glo, &ghi);
    const char *rspin = rs_opt(argc, argv, "--rsp-in", nullptr); const double minnet = atof(rs_opt(argc, argv, "--min-net-um", "0"));
    z3 *zrin = nullptr; if (rspin && !(zrin = z3_open(store_open(rspin), "rsp", nullptr))) { fprintf(stderr, "recto-sp: --rsp-in: %s\n", z3_error()); return 1; }
    z3 *zs = nullptr; double sum = 0, F = 1; const char *cache = rs_opt(argc, argv, "--cache", nullptr);
    if (!zrin) { for (int i = 4; i + 3 < argc; i++) if (!strcmp(argv[i], "--src")) { if (!(zs = z3_open(store_open(argv[i + 1]), argv[i + 2], cache))) { fprintf(stderr, "recto-sp: --src: %s\n", z3_error()); return 1; } sum = atof(argv[i + 3]); }
                 if (!zs || !(sum > 0)) { fprintf(stderr, "recto-sp: --src ROOT KEY UM (or --rsp-in DIR)\n"); return 2; } F = um / sum; }
    rr_cls *cls = nullptr; { const char *cp = rs_opt(argc, argv, "--cls", nullptr); if (cp && !(cls = rr_cls_load(cp))) { fprintf(stderr, "recto-sp: cannot load --cls %s\n", cp); return 1; } }
    const int S = atoi(rs_opt(argc, argv, "--stride", "4")); const double half = atof(rs_opt(argc, argv, "--half-um", "400"));
    const int SH = 512;
    char attrs[256]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h)\",\"h\":%d,\"recto_sp\":1}}", h);
    snprintf(path, sizeof path, "%s/sp", out); char sl[4096]; snprintf(sl, sizeof sl, "%s/sp", out); unlink(sl);
    z3w *wsp = z3w_create(path, ash, SH, 0, 0, attrs); if (!wsp) { fprintf(stderr, "recto-sp: %s\n", z3w_error()); return 1; }
    z3w *wrs = nullptr; if (!zrin) { snprintf(path, sizeof path, "%s/rsp", out); if (!(wrs = z3w_create(path, ash, SH, 0, 0, attrs))) { fprintf(stderr, "recto-sp: %s\n", z3w_error()); return 1; } }
    const int H = shp[1], W = shp[2]; const double t0 = rs_now(); long long tc = 0, te = 0, nrep = 0, nmat = 0;
    uint8_t *shard = malloc((size_t)SH * SH * SH);
    /* shard rows of the array covering the box */
    for (int64_t gz0 = (zoff[0] / SH) * SH; gz0 < zoff[0] + shp[0]; gz0 += SH) {
        const int b0 = (int)(gz0 - zoff[0] < 0 ? 0 : gz0 - zoff[0]), b1 = (int)(gz0 + SH - zoff[0] > shp[0] ? shp[0] : gz0 + SH - zoff[0]);   /* box rows in this shard row */
        const int nr = b1 - b0; if (nr <= 0) continue;
        const size_t plane = (size_t)H * W; uint8_t *net = malloc(plane * nr), *rsp = calloc(plane * nr, 1);
        { const int64_t o[3] = {zoff[0] + b0, zoff[1], zoff[2]}, nn[3] = {nr, H, W}; if (z3_read(zsp, o, nn, net, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; } }
        if (zrin) { const int64_t o[3] = {zoff[0] + b0, zoff[1], zoff[2]}, nn[3] = {nr, H, W}; if (z3_read(zrin, o, nn, rsp, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; } }
        else for (int c0 = b0; c0 < b1; c0 += 64) {
            const int c1 = c0 + 64 < b1 ? c0 + 64 : b1; const int q0 = c0 > R0 ? c0 : R0, q1 = c1 < R1 ? c1 : R1;
            if (q1 <= q0) continue;
            const int a0 = (q0 / S) * S, a1 = q1;   /* (stride blocks start at multiples of S) */
            fit_fields_skip_planes(1 | 2); fit_fields_ct_planes(0);
            const int hz = cls ? (int)ceil(half / um / h) + 1 : 0, l0 = a0 - hz < 0 ? 0 : a0 - hz, l1 = a1 + hz > shp[0] ? shp[0] : a1 + hz;   /* (--cls: the fields along the profiles: a halo of rows) */
            fit_fields_buf fb; if (fit_fields_load(in, l0, l1, 1, &fb)) { fprintf(stderr, "recto-sp: cannot load field rows %d..%d\n", l0, l1); return 1; }
            /* source block: native z [org + a0 h - half, org + a1 h + half], the box's y / x +- half */
            const double hm = half / um + 2; rs_job J = {.b = &fb, .r0 = a0 - l0, .nr = a1 - a0, .F = F, .org = {org[0] + (long long)l0 * h, org[1], org[2]}, .cls = cls, .h = h, .S = S, .um = um, .half = half,
                .prom = atof(rs_opt(argc, argv, "--prom", "25")), .sigma = atof(rs_opt(argc, argv, "--sigma", "0.5")), .sep = atof(rs_opt(argc, argv, "--sep-um", "20")), .minint = atoi(rs_opt(argc, argv, "--min-int", "3")), .H = H, .W = W};
            const int64_t *ssh = z3_meta_of(zs)->shape; const double za0 = org[0] + (double)a0 * h; const double lo[3] = {za0 - hm, org[1] - hm, org[2] - hm}, hi[3] = {za0 + (a1 - a0) * h + hm, org[1] + (double)H * h + hm, org[2] + (double)W * h + hm};
            for (int d = 0; d < 3; d++) { int64_t s0 = (int64_t)floor(lo[d] * F) - 1, s1 = (int64_t)ceil(hi[d] * F) + 2; if (s0 < 0) s0 = 0; if (s1 > ssh[d]) s1 = ssh[d]; J.so[d] = s0; J.sn[d] = s1 - s0; }
            uint8_t *sb = malloc((size_t)J.sn[0] * J.sn[1] * J.sn[2]); if (!sb) { fprintf(stderr, "recto-sp: out of memory\n"); return 1; }
            if (z3_read(zs, J.so, J.sn, sb, nt)) { fprintf(stderr, "recto-sp: source read: %s\n", z3_error()); return 1; }
            J.src = sb;
            /* output: rows a0..a1 of the box into rsp (rows from b0); the block starting before q0 / past b1 is clipped by the row test */
            uint8_t *tmpo = calloc((size_t)(a1 - a0) * plane, 1); J.out = tmpo; J.oz0 = J.r0;
            const int nyb = (H + S - 1) / S, nzb = (a1 - a0 + S - 1) / S;
            fit_par(nzb * nyb, rs_rows, &J);
            for (int r = q0; r < q1; r++) memcpy(rsp + (size_t)(r - b0) * plane, tmpo + (size_t)(r - a0) * plane, plane);
            free(tmpo); free(sb); fit_fields_free(&fb);
            tc += J.ncell; te += J.nest;
            fprintf(stderr, "\rrecto-sp: rows %d / %d (%lld cells, %.3f estimated, %.0f s)", c1, shp[0], tc, tc ? (double)te / tc : 0, rs_now() - t0);
        }
        /* blend into net */
        for (size_t i = 0; i < plane * nr; i++) {
            if (!net[i]) continue;
            nmat++;
            if (!rsp[i]) continue;
            const double r = (double)rsp[i] / net[i]; if (r < glo || r > ghi) continue;
            if (net[i] * h / 8.0 * um < minnet) continue;   /* --min-net-um: only where the network spacing is at least this wide */
            const double v = exp((1 - blend) * log(net[i]) + blend * log(rsp[i])); net[i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); nrep++;
        }
        /* write the shard row (outside the box: the input sp) */
        for (int pass = 0; pass < (wrs ? 2 : 1); pass++) {
            z3w *w = pass ? wrs : wsp; const uint8_t *src = pass ? rsp : net;
            for (int64_t sy = 0; sy * SH < ash[1]; sy++) for (int64_t sx = 0; sx * SH < ash[2]; sx++) {
                memset(shard, 0, (size_t)SH * SH * SH);
                if (!pass) { const int64_t o[3] = {gz0, sy * SH, sx * SH}; int64_t nn[3] = {SH, SH, SH}; for (int d = 0; d < 3; d++) if (o[d] + nn[d] > ash[d]) nn[d] = ash[d] - o[d];
                    uint8_t *tb = malloc((size_t)nn[0] * nn[1] * nn[2]); if (z3_read(zsp, o, nn, tb, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; }
                    for (int64_t a = 0; a < nn[0]; a++) for (int64_t b = 0; b < nn[1]; b++) memcpy(shard + ((size_t)a * SH + b) * SH, tb + ((size_t)a * nn[1] + b) * nn[2], nn[2]);
                    free(tb); }
                int any = 0;
                for (int a = 0; a < SH; a++) { const int64_t gz = gz0 + a, br = gz - zoff[0]; if (br < b0 || br >= b1) continue;
                    for (int b = 0; b < SH; b++) { const int64_t gy = sy * SH + b, by = gy - zoff[1]; if (by < 0 || by >= H) continue;
                        const int64_t gx0 = sx * SH, x0 = gx0 - zoff[2] < 0 ? zoff[2] - gx0 : 0, x1 = gx0 + SH - zoff[2] > W ? W + zoff[2] - gx0 : SH; if (x1 <= x0) continue;
                        memcpy(shard + ((size_t)a * SH + b) * SH + x0, src + ((size_t)(br - b0) * H + by) * W + (gx0 + x0 - zoff[2]), x1 - x0); any = 1; } }
                if (!any && pass) continue;
                if (z3w_write_shard(w, gz0 / SH, sy, sx, shard, nt)) { fprintf(stderr, "recto-sp: write: %s\n", z3w_error()); return 1; }
            }
        }
        free(net); free(rsp);
    }
    fprintf(stderr, "\n");
    if (z3w_close(wsp) || (wrs && z3w_close(wrs))) { fprintf(stderr, "recto-sp: close: %s\n", z3w_error()); return 1; }
    printf("recto-sp: %s -> %s: %lld cells probed, %.3f with a recto spacing; sp replaced on %.3f of the material cells (blend %g, gate %g..%g), %.0f s\n",
           in, out, tc, tc ? (double)te / tc : 0, nmat ? (double)nrep / nmat : 0, blend, glo, ghi, rs_now() - t0);
    free(shard); rr_cls_free(cls); z3_close(zsp); if (zs) z3_close(zs); if (zrin) z3_close(zrin);
    return 0;
}
