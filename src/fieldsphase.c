/* ufsm fields-phase OUT_DIR --like ROOT KEY [--ref ROOT KEY] [--conf 0.5] [--threads 16] FIELDS_DIR...
   Teacher pseudo-labels for distillation: a label phase raster (PHASE_* codes: 0..251 = phase x 252, 255 unknown; the shape and grid of
   --like, e.g. a source's 9.6 um phase raster) from the phase planes of one or more fit-field sets (ufsm predict --fit-zarr, e.g. wind96
   over the training scrolls). The field phase vector (pc, ps) is interpolated trilinearly to each voxel centre (as a vector, so the phase
   stays smooth across cells) and kept where its confidence is >= --conf; --ref keeps the human label phase wherever it is known (the
   teacher only fills the unlabelled voxels). Later field dirs fill only voxels still unknown. Only shards that overlap a field box are
   written (shard 512); the rest stays the fill (255). Output: OUT_DIR is the array (point a source's band phase {root: dirname, key:
   basename} at it). */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "store.h"
#include "zarr3.h"
#include "z3w.h"

static const char *fp_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

/* the phase vector of field set b at native voxel (z, y, x) relative to its origin, trilinear over cell centres h (i + 0.5) */
static int fp_sample(const fit_fields_buf *b, double z, double y, double x, double *c, double *s) {
    const fit_fields *f = &b->f; const double h = b->h;
    const double fz = z / h - 0.5, fy = y / h - 0.5, fx = x / h - 0.5;
    int iz = (int)floor(fz), iy = (int)floor(fy), ix = (int)floor(fx);
    const double tz = fz - iz, ty = fy - iy, tx = fx - ix;
    double sc = 0, ss = 0, sw = 0;
    for (int a = 0; a < 2; a++) for (int bb = 0; bb < 2; bb++) for (int d = 0; d < 2; d++) {
        int zz = iz + a, yy = iy + bb, xx = ix + d;
        zz = zz < 0 ? 0 : zz >= f->D ? f->D - 1 : zz; yy = yy < 0 ? 0 : yy >= f->H ? f->H - 1 : yy; xx = xx < 0 ? 0 : xx >= f->W ? f->W - 1 : xx;
        const double w = (a ? tz : 1 - tz) * (bb ? ty : 1 - ty) * (d ? tx : 1 - tx);
        const size_t i = ((size_t)zz * f->H + yy) * f->W + xx;
        if (f->pc[i] == 128 && f->ps[i] == 128) continue;
        sc += w * (f->pc[i] - 128.0) / 127.0; ss += w * (f->ps[i] - 128.0) / 127.0; sw += w;
    }
    if (sw < 0.5) return 0;
    *c = sc / sw; *s = ss / sw; return 1;
}

typedef struct {
    const fit_fields_buf *B; int nd; z3 *zr; z3w *w; const int64_t *shape; int S; double cmin; const int64_t *list; int64_t nl;
    _Atomic int64_t next; _Atomic long long nref, nteach, nlow; _Atomic int fail;
} fp_job;
static void *fp_worker(void *a) {
    fp_job *J = a; const int S = J->S; uint8_t *buf = malloc((size_t)S * S * S);
    for (int64_t k; (k = atomic_fetch_add(&J->next, 1)) < J->nl;) {
        const int64_t sz = J->list[3 * k], sy = J->list[3 * k + 1], sx = J->list[3 * k + 2], o[3] = {sz * S, sy * S, sx * S};
        int64_t n[3]; for (int a2 = 0; a2 < 3; a2++) n[a2] = o[a2] + S <= J->shape[a2] ? S : J->shape[a2] - o[a2];
        memset(buf, 255, (size_t)S * S * S); long long nref = 0, nteach = 0, nlow = 0;
        if (J->zr) {   /* the human labels, packed n[0] x n[1] x n[2], spread into the S^3 shard layout */
            uint8_t *t = malloc((size_t)n[0] * n[1] * n[2]);
            if (!t || z3_read(J->zr, o, n, t, 1)) { J->fail = 1; free(t); continue; }
            for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) memcpy(buf + ((size_t)z * S + y) * S, t + ((size_t)z * n[1] + y) * n[2], n[2]);
            free(t);
        }
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
            uint8_t *v = buf + ((size_t)z * S + y) * S + x; if (*v <= 251) { nref++; continue; }
            for (int d = 0; d < J->nd; d++) {
                const fit_fields_buf *b = &J->B[d]; const long long *bo = b->origin;
                const double lz = o[0] + z - bo[0] + 0.5, ly = o[1] + y - bo[1] + 0.5, lx = o[2] + x - bo[2] + 0.5;
                if (lz < 0 || ly < 0 || lx < 0 || lz >= (double)b->f.D * b->h || ly >= (double)b->f.H * b->h || lx >= (double)b->f.W * b->h) continue;
                double c, s; if (!fp_sample(b, lz, ly, lx, &c, &s)) continue;
                if (sqrt(c * c + s * s) < J->cmin) { nlow++; continue; }
                double ph = atan2(s, c) / (2 * M_PI); ph -= floor(ph);
                int code = (int)lround(ph * 252); if (code >= 252) code = 0;
                *v = (uint8_t)code; nteach++; break;
            }
        }
        if (z3w_write_shard(J->w, sz, sy, sx, buf, 1)) J->fail = 1;
        J->nref += nref; J->nteach += nteach; J->nlow += nlow;
    }
    free(buf); return nullptr;
}

int cmd_fields_phase(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fields-phase OUT_DIR --like ROOT KEY [--ref ROOT KEY] [--conf 0.5] [--threads 16] FIELDS_DIR...\n"); return 2; }
    const char *out = argv[2], *lr = nullptr, *lk = nullptr, *rr = nullptr, *rk = nullptr; const char *dirs[256]; int nd = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--like") && i + 2 < argc) { lr = argv[i + 1]; lk = argv[i + 2]; i += 2; }
        else if (!strcmp(argv[i], "--ref") && i + 2 < argc) { rr = argv[i + 1]; rk = argv[i + 2]; i += 2; }
        else if ((!strcmp(argv[i], "--conf") || !strcmp(argv[i], "--threads")) && i + 1 < argc) i++;
        else if (argv[i][0] != '-' && nd < 256) dirs[nd++] = argv[i];
    }
    if (!lr || !nd) { fprintf(stderr, "fields-phase: --like ROOT KEY and at least one FIELDS_DIR\n"); return 2; }
    const double cmin = atof(fp_opt(argc, argv, "--conf", "0.5")); const int nt = atoi(fp_opt(argc, argv, "--threads", "16"));
    z3 *zl = z3_open(store_open(lr), lk, nullptr); if (!zl) { fprintf(stderr, "fields-phase: %s\n", z3_error()); return 1; }
    const int64_t *shape = z3_meta_of(zl)->shape;
    z3 *zr = nullptr; if (rr && !(zr = z3_open(store_open(rr), rk, nullptr))) { fprintf(stderr, "fields-phase: %s\n", z3_error()); return 1; }
    fit_fields_skip_planes(1 | 2 | 4 | 8); fit_fields_ct_planes(0);
    fit_fields_buf *B = calloc(nd, sizeof *B);
    for (int d = 0; d < nd; d++) if (fit_fields_load(dirs[d], 0, 0, 1, &B[d])) { fprintf(stderr, "fields-phase: cannot load %s\n", dirs[d]); return 1; }
    const int S = 512; const int64_t g[3] = {(shape[0] + S - 1) / S, (shape[1] + S - 1) / S, (shape[2] + S - 1) / S};
    char at[256]; snprintf(at, sizeof at, "{\"ufsm\": {\"content\": \"fields-phase teacher pseudo-labels\", \"conf\": %g, \"ref\": %s}}", cmin, zr ? "true" : "false");
    z3w *w = z3w_create(out, shape, S, 0.f, 255, at); if (!w) { fprintf(stderr, "fields-phase: %s\n", z3w_error()); return 1; }
    /* the shards that overlap a field box, then one worker per thread (own S^3 buffer; z3w_write_shard is thread-safe across shards) */
    int64_t *list = malloc(sizeof(int64_t) * 3 * (size_t)(g[0] * g[1] * g[2])); int64_t nl = 0;
    for (int64_t sz = 0; sz < g[0]; sz++) for (int64_t sy = 0; sy < g[1]; sy++) for (int64_t sx = 0; sx < g[2]; sx++) {
        const int64_t o[3] = {sz * S, sy * S, sx * S}; int hit = 0;
        for (int d = 0; d < nd && !hit; d++) {
            const long long *bo = B[d].origin; const int64_t be[3] = {bo[0] + (int64_t)B[d].f.D * B[d].h, bo[1] + (int64_t)B[d].f.H * B[d].h, bo[2] + (int64_t)B[d].f.W * B[d].h};
            hit = o[0] < be[0] && o[0] + S > bo[0] && o[1] < be[1] && o[1] + S > bo[1] && o[2] < be[2] && o[2] + S > bo[2];
        }
        if (hit) { list[3 * nl] = sz; list[3 * nl + 1] = sy; list[3 * nl + 2] = sx; nl++; }
    }
    fp_job J = {.B = B, .nd = nd, .zr = zr, .w = w, .shape = shape, .S = S, .cmin = cmin, .list = list, .nl = nl};
    const int nw = nt < 1 ? 1 : nt > 32 ? 32 : nt; pthread_t th[32];
    for (int i = 0; i < nw; i++) pthread_create(&th[i], nullptr, fp_worker, &J);
    for (int i = 0; i < nw; i++) pthread_join(th[i], nullptr);
    free(list);
    if (J.fail) { fprintf(stderr, "fields-phase: a shard failed (read or write)\n"); return 1; }
    const long long nsh = nl, nref = J.nref, nteach = J.nteach, nlow = J.nlow;
    z3w_close(w);
    printf("fields-phase: %lld shards, voxels: %lld human labels kept, %lld teacher labels (conf >= %g), %lld teacher voxels below conf -> %s\n", nsh, nref, nteach, cmin, nlow, out);
    for (int d = 0; d < nd; d++) fit_fields_free(&B[d]);
    free(B);
    return 0;
}
