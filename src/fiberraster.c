/* ufsm fiber-raster <fibres.afv | fibres.fib> OUT --um U (--shape Z,Y,X | --ct ROOT KEY [--iso 64]) [--level 0] [--tr T.json]
       [--zband Z0,Z1] [--radius 2] [--guard 3] [--shard 512] [--threads 4]
   Training labels for fibre-family mask heads from fibre polylines (an AFV, or UFIB level-0 points): two prob254 volumes on the level-L
   CT grid (U = that grid's voxel size), OUT/H/<U> and OUT/V/<U> (zarr v3, lossless, fill 255 = unknown), usable as the "recto" and
   "sheet" targets of a sources entry ({"root": OUT/H, "encoding": "prob254"}).
   Per voxel, with dH / dV its distance to the nearest H / V fibre axis (level voxels):
     plane H: 254 where dH <= radius and dV > radius; 0 where dV <= radius and dH > radius (the other family: the contrast the head has
              to learn); 0 where the CT is below --iso (air; only with --ct); 255 (no loss) elsewhere - within radius of both, in the
              guard ring radius .. radius + guard about any fibre, near a fibre without a family, and in all papyrus no tracing covers
              (the published tracings hold fibres > 1 mm only, 11-13 % of the papyrus within 4 voxels: untraced is not "no fibre");
     plane V: the same with H and V exchanged.
   Shards (--shard^3) are independent; --zband limits the z range written (level voxels, widened to whole shards). CPU. */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"

static const char *fr_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int fr_opti(int argc, char **argv, const char *k) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return i; return 0; }
typedef struct {
    afv *A; const fib_set *F; double (*fbb)[6]; pthread_mutex_t mu; int has_tr; double M[3][4], Mi[3][3]; double sc;
    int64_t shape[3]; int S, iso; double rad, guard; z3 *ct; z3w *w[2];
    int64_t s0[3], ns[3]; atomic_long next; atomic_int fail; atomic_llong cnt[4];   /* H 254, H 0, V 254, V 0 (voxels) */
} fr_job;
/* distance code: 16 x voxels, 255 = farther than radius + guard */
static void fr_stamp(uint8_t *d, const int64_t so[3], const int64_t n[3], const double p[3], double R) {
    const int64_t lo[3] = {(int64_t)ceil(p[0] - R), (int64_t)ceil(p[1] - R), (int64_t)ceil(p[2] - R)}, hi[3] = {(int64_t)floor(p[0] + R), (int64_t)floor(p[1] + R), (int64_t)floor(p[2] + R)};
    for (int64_t z = lo[0] < so[0] ? so[0] : lo[0]; z <= hi[0] && z < so[0] + n[0]; z++) for (int64_t y = lo[1] < so[1] ? so[1] : lo[1]; y <= hi[1] && y < so[1] + n[1]; y++) {
        uint8_t *row = d + ((size_t)(z - so[0]) * n[1] + (size_t)(y - so[1])) * n[2]; const double dz = z - p[0], dy = y - p[1], a = dz * dz + dy * dy;
        for (int64_t x = lo[2] < so[2] ? so[2] : lo[2]; x <= hi[2] && x < so[2] + n[2]; x++) { const double dx = x - p[2], dd = a + dx * dx; if (dd > R * R) continue;
            const int c = (int)(16 * sqrt(dd)); if (c < row[x - so[2]]) row[x - so[2]] = (uint8_t)c; }
    }
}
static void *fr_worker(void *arg) {
    fr_job *J = arg; const int S = J->S; const size_t nv = (size_t)S * S * S; const double R = J->rad + J->guard;
    uint8_t *d[3], *o[2], *ct = J->ct ? malloc(nv) : nullptr; for (int k = 0; k < 3; k++) d[k] = malloc(nv); for (int k = 0; k < 2; k++) o[k] = malloc(nv);
    const long total = (long)(J->ns[0] * J->ns[1] * J->ns[2]);
    for (long t; (t = atomic_fetch_add(&J->next, 1)) < total && !atomic_load(&J->fail);) {
        const int64_t sh[3] = {J->s0[0] + t / (J->ns[1] * J->ns[2]), J->s0[1] + t / J->ns[2] % J->ns[1], J->s0[2] + t % J->ns[2]}, so[3] = {sh[0] * S, sh[1] * S, sh[2] * S}, n[3] = {S, S, S};
        const double lo[3] = {(double)so[0] - R, (double)so[1] - R, (double)so[2] - R}, hi[3] = {(double)(so[0] + S) + R, (double)(so[1] + S) + R, (double)(so[2] + S) + R};
        afv_run *Rn = nullptr; int nr = 0, own = 0;
        if (J->A) {
            double qlo[3] = {1e30, 1e30, 1e30}, qhi[3] = {-1e30, -1e30, -1e30};
            for (int c = 0; c < 8; c++) { const double g[3] = {((c & 1 ? hi[2] : lo[2]) + 0.5) * J->sc - 0.5, ((c & 2 ? hi[1] : lo[1]) + 0.5) * J->sc - 0.5, ((c & 4 ? hi[0] : lo[0]) + 0.5) * J->sc - 0.5}; double a[3];
                for (int k = 0; k < 3; k++) a[k] = J->has_tr ? J->M[k][0] * g[0] + J->M[k][1] * g[1] + J->M[k][2] * g[2] + J->M[k][3] : g[k];
                for (int k = 0; k < 3; k++) { if (a[k] < qlo[k]) qlo[k] = a[k]; if (a[k] > qhi[k]) qhi[k] = a[k]; } }
            pthread_mutex_lock(&J->mu); nr = afv_query(J->A, qlo, qhi, &Rn); pthread_mutex_unlock(&J->mu);
            if (nr < 0) { atomic_store(&J->fail, 1); break; }
            own = 1;
        } else {
            Rn = calloc((size_t)J->F->n + 1, sizeof *Rn);
            for (int r = 0; r < J->F->n; r++) if (J->fbb[r][3] >= lo[0] && J->fbb[r][0] < hi[0] && J->fbb[r][4] >= lo[1] && J->fbb[r][1] < hi[1] && J->fbb[r][5] >= lo[2] && J->fbb[r][2] < hi[2]) {
                Rn[nr].fiber = r; Rn[nr].family = J->F->fam[r]; Rn[nr].n = J->F->off[r + 1] - J->F->off[r]; Rn[nr].xyz = nullptr; nr++; }
        }
        for (int k = 0; k < 3; k++) memset(d[k], 255, nv);
        long stamped = 0;
        for (int r = 0; r < nr; r++) {
            const afv_run *q = &Rn[r]; uint8_t *dd = d[q->family == 'H' ? 0 : q->family == 'V' ? 1 : 2]; double prev[3] = {0, 0, 0};
            for (int i = 0; i < q->n; i++) {
                double p[3];
                if (own) { double g[3] = {q->xyz[3 * i], q->xyz[3 * i + 1], q->xyz[3 * i + 2]};
                    if (J->has_tr) { const double u[3] = {g[0] - J->M[0][3], g[1] - J->M[1][3], g[2] - J->M[2][3]}; for (int k = 0; k < 3; k++) g[k] = J->Mi[k][0] * u[0] + J->Mi[k][1] * u[1] + J->Mi[k][2] * u[2]; }
                    for (int k = 0; k < 3; k++) p[k] = (g[2 - k] + 0.5) / J->sc - 0.5;
                } else { const float *g = J->F->pts + 3 * (size_t)(J->F->off[q->fiber] + i); for (int k = 0; k < 3; k++) p[k] = (g[k] + 0.5) / J->sc - 0.5; }
                /* the segment prev -> p in steps of <= 1 voxel */
                const double len = i ? sqrt(pow(p[0] - prev[0], 2) + pow(p[1] - prev[1], 2) + pow(p[2] - prev[2], 2)) : 0; const int m = i ? (int)ceil(len) : 1;
                for (int k = 1; k <= m; k++) { const double f = (double)k / m, s[3] = {prev[0] + f * (p[0] - prev[0]), prev[1] + f * (p[1] - prev[1]), prev[2] + f * (p[2] - prev[2])};
                    const double *u = i ? s : p; if (u[0] < lo[0] || u[0] > hi[0] || u[1] < lo[1] || u[1] > hi[1] || u[2] < lo[2] || u[2] > hi[2]) continue;
                    fr_stamp(dd, so, n, u, R); stamped++; }
                for (int k = 0; k < 3; k++) prev[k] = p[k];
            }
        }
        if (own) afv_runs_free(Rn, nr); else free(Rn);
        int have_ct = 0;
        if (J->ct) { int64_t cn[3]; for (int k = 0; k < 3; k++) cn[k] = so[k] + S > J->shape[k] ? J->shape[k] - so[k] : S;
            uint8_t *tmp = malloc((size_t)cn[0] * cn[1] * cn[2]);
            if (!tmp || z3_read(J->ct, so, cn, tmp, 1)) { fprintf(stderr, "fiber-raster: CT read failed\n"); atomic_store(&J->fail, 1); free(tmp); break; }
            memset(ct, 255, nv); for (int64_t z = 0; z < cn[0]; z++) for (int64_t y = 0; y < cn[1]; y++) memcpy(ct + ((size_t)z * S + y) * S, tmp + ((size_t)z * cn[1] + y) * cn[2], (size_t)cn[2]);
            free(tmp); have_ct = 1; }
        if (!stamped && !have_ct) continue;   /* all unknown: the shard is not written (fill 255) */
        const int cr = (int)(16 * J->rad); long long c[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < nv; i++) {
            const int h = d[0][i] <= cr, v = d[1][i] <= cr, any = d[0][i] != 255 || d[1][i] != 255 || d[2][i] != 255, air = have_ct && ct[i] < J->iso;
            o[0][i] = h && !v && d[2][i] > cr ? 254 : v && !h && d[2][i] > cr ? 0 : !any && air ? 0 : 255;
            o[1][i] = v && !h && d[2][i] > cr ? 254 : h && !v && d[2][i] > cr ? 0 : !any && air ? 0 : 255;
            c[0] += o[0][i] == 254; c[1] += o[0][i] == 0; c[2] += o[1][i] == 254; c[3] += o[1][i] == 0;
        }
        for (int k = 0; k < 4; k++) atomic_fetch_add(&J->cnt[k], c[k]);
        for (int k = 0; k < 2; k++) if (z3w_write_shard(J->w[k], sh[0], sh[1], sh[2], o[k], 1)) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); atomic_store(&J->fail, 1); }
    }
    for (int k = 0; k < 3; k++) free(d[k]);
    for (int k = 0; k < 2; k++) free(o[k]);
    free(ct);
    return nullptr;
}

int cmd_fiber_raster(int argc, char **argv) {
    fr_job J = {0}; const double um = atof(fr_opt(argc, argv, "--um", "0")); long long sh[3] = {0, 0, 0}; const int ci = fr_opti(argc, argv, "--ct");
    const int has_shape = sscanf(fr_opt(argc, argv, "--shape", ""), "%lld,%lld,%lld", &sh[0], &sh[1], &sh[2]) == 3;
    if (argc < 6 || !(um > 0) || (!has_shape && !ci)) {
        fprintf(stderr, "usage: ufsm fiber-raster <fibres.afv|.fib> OUT --um U (--shape Z,Y,X | --ct ROOT KEY [--iso 64]) [--level 0] [--tr T.json] [--zband Z0,Z1]\n"
                        "       [--radius 2] [--guard 3] [--shard 512] [--threads 4]   H / V fibre mask labels (prob254: OUT/H/<U>, OUT/V/<U>)\n");
        return 2;
    }
    const int level = atoi(fr_opt(argc, argv, "--level", "0")), nth = atoi(fr_opt(argc, argv, "--threads", "4")); J.S = atoi(fr_opt(argc, argv, "--shard", "512"));
    J.sc = ldexp(1.0, level); J.rad = atof(fr_opt(argc, argv, "--radius", "2")); J.guard = atof(fr_opt(argc, argv, "--guard", "3")); J.iso = atoi(fr_opt(argc, argv, "--iso", "64"));
    if (J.S < 128 || J.S % 128 || !(J.rad > 0) || J.guard < 0 || J.rad + J.guard > 15 || nth < 1 || level < 0 || level > 8) { fprintf(stderr, "fiber-raster: bad options (--shard a multiple of 128, radius + guard <= 15)\n"); return 2; }
    if (ci) {
        if (ci + 2 >= argc) { fprintf(stderr, "fiber-raster: --ct ROOT KEY\n"); return 2; }
        store *s = store_open(argv[ci + 1]); J.ct = s ? z3_open(s, argv[ci + 2], nullptr) : nullptr; if (!J.ct) { fprintf(stderr, "fiber-raster: cannot open the CT %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
        for (int k = 0; k < 3; k++) { const int64_t v = z3_meta_of(J.ct)->shape[k]; if (has_shape && sh[k] != v) { fprintf(stderr, "fiber-raster: --shape differs from the CT's %lld %lld %lld\n", (long long)z3_meta_of(J.ct)->shape[0], (long long)z3_meta_of(J.ct)->shape[1], (long long)z3_meta_of(J.ct)->shape[2]); return 1; } sh[k] = v; }
    }
    for (int k = 0; k < 3; k++) { J.shape[k] = sh[k]; if (sh[k] < 1) { fprintf(stderr, "fiber-raster: bad shape\n"); return 2; } }
    const char *trp = fr_opt(argc, argv, "--tr", nullptr); J.has_tr = trp != nullptr;
    if (trp && fc_transform(trp, J.M, J.Mi)) { fprintf(stderr, "fiber-raster: cannot read the transform %s\n", trp); return 1; }
    fib_set F = {0};
    { FILE *f = fopen(argv[2], "rb"); char mg[4] = ""; if (!f || fread(mg, 1, 4, f) != 4) { fprintf(stderr, "fiber-raster: cannot read %s\n", argv[2]); return 1; } fclose(f);
      if (!memcmp(mg, "UFIB", 4)) {
          if (fib_load(argv[2], &F)) { fprintf(stderr, "fiber-raster: bad UFIB file %s\n", argv[2]); return 1; }
          J.F = &F; J.fbb = malloc(((size_t)F.n + 1) * sizeof *J.fbb);
          for (int r = 0; r < F.n; r++) { for (int k = 0; k < 3; k++) { J.fbb[r][k] = 1e30; J.fbb[r][3 + k] = -1e30; }
              for (int i = F.off[r]; i < F.off[r + 1]; i++) for (int k = 0; k < 3; k++) { const double v = (F.pts[3 * (size_t)i + k] + 0.5) / J.sc - 0.5; if (v < J.fbb[r][k]) J.fbb[r][k] = v; if (v > J.fbb[r][3 + k]) J.fbb[r][3 + k] = v; } }
      } else if (!(J.A = afv_open(argv[2]))) return 1; }
    long long zb[2] = {0, sh[0]}; if (*fr_opt(argc, argv, "--zband", "") && sscanf(fr_opt(argc, argv, "--zband", ""), "%lld,%lld", &zb[0], &zb[1]) != 2) { fprintf(stderr, "fiber-raster: --zband Z0,Z1\n"); return 2; }
    if (zb[0] < 0) zb[0] = 0;
    if (zb[1] > sh[0]) zb[1] = sh[0];
    if (zb[1] <= zb[0]) { fprintf(stderr, "fiber-raster: empty --zband\n"); return 2; }
    J.s0[0] = zb[0] / J.S; J.ns[0] = (zb[1] + J.S - 1) / J.S - J.s0[0]; for (int k = 1; k < 3; k++) { J.s0[k] = 0; J.ns[k] = (J.shape[k] + J.S - 1) / J.S; }
    char lv[64], p[2200], at[600]; z3w_level_name(um, lv, sizeof lv);
    for (int k = 0; k < 2; k++) {
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"fiber-raster %c\",\"encoding\":\"prob254: 254 = a %c fibre, 0 = a %c fibre%s, 255 = unknown\",\"radius\":%g,\"guard\":%g,\"zband\":[%lld,%lld]}}",
                 "HV"[k], "HV"[k], "VH"[k], J.ct ? " or air" : "", J.rad, J.guard, zb[0], zb[1]);
        snprintf(p, sizeof p, "mkdir -p '%s/%c'", argv[3], "HV"[k]); if (system(p)) return 1;
        snprintf(p, sizeof p, "%s/%c", argv[3], "HV"[k]); if (z3w_write_group(p, &um, 1, "fiber-raster", at)) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); return 1; }
        snprintf(p, sizeof p, "%s/%c/%s", argv[3], "HV"[k], lv); J.w[k] = z3w_create(p, J.shape, J.S, 0, 255, at); if (!J.w[k]) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); return 1; }
    }
    pthread_mutex_init(&J.mu, nullptr); pthread_t th[64]; const int nt = nth > 64 ? 64 : nth;
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, fr_worker, &J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    int bad = atomic_load(&J.fail); for (int k = 0; k < 2; k++) bad |= z3w_close(J.w[k]) != 0;
    if (bad) { fprintf(stderr, "fiber-raster: FAILED\n"); return 1; }
    printf("fiber-raster: %lld x %lld x %lld level-%d voxels of %g um, z %lld..%lld, radius %g guard %g: H plane %lld fibre / %lld other voxels, V plane %lld / %lld -> %s/{H,V}/%s\n",
           sh[0], sh[1], sh[2], level, um, zb[0], zb[1], J.rad, J.guard, (long long)atomic_load(&J.cnt[0]), (long long)atomic_load(&J.cnt[1]), (long long)atomic_load(&J.cnt[2]), (long long)atomic_load(&J.cnt[3]), argv[3], lv);
    if (J.A) afv_close(J.A);
    fib_free(&F); free(J.fbb); if (J.ct) z3_close(J.ct);
    return 0;
}
