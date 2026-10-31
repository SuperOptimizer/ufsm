/* ufsm ink-blend (see cmd_ink_blend) */
#include "sysinfo.h"
#include "fblend.h"
#include "fit.h"
#include "pyramid.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* ================= ink-blend ================= */
/* ufsm ink-blend OUT --um U --tile SZ,SY,SX --grids z,y,x:z,y,x:... IN1 IN2 ... [--levels 4] [--shard 512] [--threads 16]: overlapping
   whole-volume predicts of one model (predict --grid-origin, window - 2 halo = the tile) blended voxel by voxel: weight of input i = the
   product over the axes of sin^2 of the voxel's place in its tile (fb_tile_w, as field-blend --tile), so two grids offset by half a tile
   on each axis sum to 1 and every tile border gets weight ~0. Inputs: uint8 probability groups on one grid (level named by U). Output:
   uint8 probability (volcomp --q, default 32 as predict; 0 lossless) with OUT's coarse levels mean-pooled. Shards where every input is absent are skipped (fill 0). */
typedef struct { uint8_t **in; int n; const float *wz, *wy, *wx; int S; int64_t n3[3]; uint8_t *out; } ib_job;
static void ib_rows(void *a, int z0, int z1, int t) {
    (void)t; const ib_job *J = a; const int64_t ny = J->n3[1], nx = J->n3[2];
    for (int z = z0; z < z1; z++) for (int64_t y = 0; y < ny; y++) {
        const size_t r = ((size_t)z * ny + y) * nx;
        for (int64_t x = 0; x < nx; x++) {
            double s = 0, w = 0;
            for (int i = 0; i < J->n; i++) {
                const double wi = (double)J->wz[(size_t)i * J->S + z] * J->wy[(size_t)i * J->S + y] * J->wx[(size_t)i * J->S + x];
                s += wi * J->in[i][r + x]; w += wi;
            }
            J->out[r + x] = w > 0 ? (uint8_t)fmin(255.0, s / w + 0.5) : 0;
        }
    }
}
int cmd_ink_blend(int argc, char **argv) {
    const char *in[32]; int n = 0; const char *out = nullptr, *tile = nullptr, *grids = nullptr, *origins = nullptr;
    double um = 0, q = 32; int nlev = 4, S = 512, nt = ufsm_threads(0, 16);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--um") && i + 1 < argc) { um = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--tile") && i + 1 < argc) { tile = argv[++i]; continue; }
        if (!strcmp(argv[i], "--grids") && i + 1 < argc) { grids = argv[++i]; continue; }
        if (!strcmp(argv[i], "--origins") && i + 1 < argc) { origins = argv[++i]; continue; }
        if (!strcmp(argv[i], "--levels") && i + 1 < argc) { nlev = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--shard") && i + 1 < argc) { S = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--q") && i + 1 < argc) { q = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) { nt = atoi(argv[++i]); continue; }
        if (!strncmp(argv[i], "--", 2)) { fprintf(stderr, "ink-blend: unknown option %s\n", argv[i]); return 2; }
        if (!out) out = argv[i]; else if (n < 32) in[n++] = argv[i]; else { fprintf(stderr, "ink-blend: at most 32 inputs\n"); return 2; }
    }
    double st[3], g[32][3];
    if (!out || n < 1 || um <= 0 || !tile || !grids || sscanf(tile, "%lf,%lf,%lf", &st[0], &st[1], &st[2]) != 3 || !(st[0] > 0 && st[1] > 0 && st[2] > 0) || S < 128 || S % 128) {
        fprintf(stderr, "usage: ufsm ink-blend <out> --um U --tile SZ,SY,SX --grids z,y,x:z,y,x:... <in1> <in2> ... [--origins z,y,x:...] [--levels 4] [--shard 512] [--q 32] [--threads 16]\n"); return 2; }
    { int k = 0; char *t = strdup(grids);
      for (char *q = strtok(t, ":"); q; q = strtok(nullptr, ":")) { if (k < 32 && sscanf(q, "%lf,%lf,%lf", &g[k][0], &g[k][1], &g[k][2]) != 3) { free(t); fprintf(stderr, "ink-blend: --grids z,y,x:...\n"); return 2; } k++; }
      free(t); if (k != n) { fprintf(stderr, "ink-blend: %d grids for %d inputs\n", k, n); return 2; } }
    int64_t org[32][3] = {{0}};   /* --origins: output voxel = input i voxel + origin i (predicts of a --box without --full-grid); outside an input its weight is 0 */
    if (origins) { int k = 0; char *t = strdup(origins);
      for (char *q = strtok(t, ":"); q; q = strtok(nullptr, ":")) { long long a, b, c; if (k < 32 && sscanf(q, "%lld,%lld,%lld", &a, &b, &c) != 3) { free(t); fprintf(stderr, "ink-blend: --origins z,y,x:...\n"); return 2; } if (k < 32) { org[k][0] = a; org[k][1] = b; org[k][2] = c; } k++; }
      free(t); if (k != n) { fprintf(stderr, "ink-blend: %d origins for %d inputs\n", k, n); return 2; } }
    char lv[32]; z3w_level_name(um, lv, sizeof lv);
    store *ss[32]; z3 *zz[32]; int64_t shape[3] = {0, 0, 0}, ish[32][3];
    for (int i = 0; i < n; i++) {
        ss[i] = store_open(in[i]); zz[i] = ss[i] ? z3_open(ss[i], lv, nullptr) : nullptr;
        if (!zz[i]) { fprintf(stderr, "ink-blend: cannot open %s/%s: %s\n", in[i], lv, z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(zz[i]);
        for (int d = 0; d < 3; d++) { ish[i][d] = m->shape[d]; if (org[i][d] + m->shape[d] > shape[d]) shape[d] = org[i][d] + m->shape[d]; }   /* the output covers every input */
    }
    const char *attrs = "{\"ufsm\":{\"content\":\"ink probability (ink-blend of overlapping predicts)\"}}";
    char odir[1400]; snprintf(odir, sizeof odir, "%s/%s", out, lv);
    z3w *w = z3w_create(odir, shape, S, (float)q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    const size_t sv = (size_t)S * S * S;
    uint8_t *buf[32]; for (int i = 0; i < n; i++) if (!(buf[i] = malloc(sv))) { fprintf(stderr, "ink-blend: out of memory\n"); return 1; }
    uint8_t *ob = malloc(sv); float *wv[3]; for (int d = 0; d < 3; d++) wv[d] = malloc((size_t)n * S * sizeof(float));
    int64_t ns[3]; for (int d = 0; d < 3; d++) ns[d] = (shape[d] + S - 1) / S;
    int nw = 0, nskip = 0; double t0 = now();
    for (int64_t a = 0; a < ns[0]; a++) for (int64_t b = 0; b < ns[1]; b++) for (int64_t c = 0; c < ns[2]; c++) {
        const int64_t o[3] = {a * S, b * S, c * S}, nn[3] = {S, S, S};
        int any = 0;
        for (int i = 0; i < n && !any; i++) {   /* any input shard holding this output shard's box (input-local voxels o - origin, clipped) */
            const z3_meta *m = z3_meta_of(zz[i]); int64_t lo[3], hi[3]; int ok = 1;
            for (int d = 0; d < 3; d++) { lo[d] = o[d] - org[i][d]; hi[d] = lo[d] + S - 1; if (lo[d] < 0) lo[d] = 0; if (hi[d] > ish[i][d] - 1) hi[d] = ish[i][d] - 1; if (lo[d] > hi[d]) ok = 0; }
            if (!ok) continue;
            for (int64_t za = lo[0] / m->shard[0]; za <= hi[0] / m->shard[0] && !any; za++)
                for (int64_t ya = lo[1] / m->shard[1]; ya <= hi[1] / m->shard[1] && !any; ya++)
                    for (int64_t xa = lo[2] / m->shard[2]; xa <= hi[2] / m->shard[2] && !any; xa++)
                        if (z3_shard_present(zz[i], za, ya, xa) == 1) any = 1;
        }
        if (!any) { nskip++; continue; }
        for (int i = 0; i < n; i++) {
            const int64_t lo[3] = {o[0] - org[i][0], o[1] - org[i][1], o[2] - org[i][2]};
            if (z3_read(zz[i], lo, nn, buf[i], nt)) { fprintf(stderr, "ink-blend: read %s: %s\n", in[i], z3_error()); return 1; }
            for (int d = 0; d < 3; d++) for (int k = 0; k < S; k++)   /* outside input i: weight 0 */
                wv[d][(size_t)i * S + k] = lo[d] + k < 0 || lo[d] + k >= ish[i][d] ? 0.f : (float)fb_tile_w(o[d], 1, k, st[d], g[i][d]);
        }
        ib_job J = {buf, n, wv[0], wv[1], wv[2], S, {S, S, S}, ob};
        fit_par(S, ib_rows, &J);
        if (z3w_write_shard(w, a, b, c, ob, nt)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        if (++nw % 50 == 0) fprintf(stderr, "ink-blend: %d shards (%d skipped) %.0f s\n", nw, nskip, now() - t0);
    }
    z3w_close(w);
    fprintf(stderr, "ink-blend: %d shards written, %d empty, %.0f s\n", nw, nskip, now() - t0);
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um, l, shape, S, (float)q, 0, nt, attrs)) return 1;
    pyramid_write_group(out, um, nlev, "ink-blend", attrs);
    for (int i = 0; i < n; i++) { free(buf[i]); z3_close(zz[i]); store_close(ss[i]); }
    free(ob); for (int d = 0; d < 3; d++) free(wv[d]);
    return 0;
}
