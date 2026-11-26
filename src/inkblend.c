/* ufsm ink-blend (see cmd_ink_blend) */
#include "sysinfo.h"
#include "optcheck.h"
#include "fblend.h"
#include "fit.h"
#include "inkblend.h"
#include "pyramid.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
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
   uint8 probability (volcomp --q, default 0 = lossless since 2026-11-14; was 32) with OUT's coarse levels mean-pooled. Shards where every input is absent are skipped (fill 0). */
/* mk[i]: nullptr or 0 = absent. bp (block mode, perf 11-25): per B^3 block of the shard (nb per axis) and input, 1 = present, instead of
   mk: a block no input is present in is written 0 without its arithmetic (the full pass gives it 0: every weight masked), the rest as
   with the masks (the same flags, weights and order) */
typedef struct { uint8_t **in; int n; const float *wz, *wy, *wx; int S; int64_t n3[3]; uint8_t *out; uint8_t *const *mk; const uint8_t *bp; int B, nb; } ib_job;
static void ib_seg(const ib_job *J, int z, int64_t y, int64_t x0, int64_t x1, const uint8_t *p);
static void ib_rows(void *a, int z0, int z1, int t) {
    (void)t; const ib_job *J = a; const int64_t ny = J->n3[1], nx = J->n3[2];
    if (J->bp) {
        for (int z = z0; z < z1; z++) for (int64_t y = 0; y < ny; y++)
            for (int bx = 0; bx < J->nb; bx++) {
                const uint8_t *p = J->bp + ((size_t)((z / J->B) * J->nb + y / J->B) * J->nb + bx) * J->n; int any = 0;
                for (int i = 0; i < J->n; i++) any |= p[i];
                if (any) ib_seg(J, z, y, (int64_t)bx * J->B, (int64_t)(bx + 1) * J->B, p);
                else memset(J->out + ((size_t)z * ny + y) * nx + (size_t)bx * J->B, 0, (size_t)J->B);
            }
        return;
    }
    for (int z = z0; z < z1; z++) for (int64_t y = 0; y < ny; y++) {
        const size_t r = ((size_t)z * ny + y) * nx;
        if (J->n == 2) {   /* (the unroll's two grids) the same arithmetic in the same order: wz wy per row, then x wx */
            volatile double wzy0v = (double)J->wz[z] * J->wy[y], wzy1v = (double)J->wz[(size_t)J->S + z] * J->wy[(size_t)J->S + y];
            const double wzy0 = wzy0v, wzy1 = wzy1v;
            const float *wx0 = J->wx, *wx1 = J->wx + J->S; const uint8_t *i0 = J->in[0] + r, *i1 = J->in[1] + r;
            const uint8_t *m0 = J->mk[0] ? J->mk[0] + r : nullptr, *m1 = J->mk[1] ? J->mk[1] + r : nullptr; uint8_t *o = J->out + r;
            for (int64_t x = 0; x < nx; x++) {
                double s = 0, w = 0;
                if (!m0 || m0[x]) { const double wi = wzy0 * wx0[x]; s += wi * i0[x]; w += wi; }
                if (!m1 || m1[x]) { const double wi = wzy1 * wx1[x]; s += wi * i1[x]; w += wi; }
                o[x] = w > 0 ? (uint8_t)fmin(255.0, s / w + 0.5) : 0;
            }
            continue;
        }
        for (int64_t x = 0; x < nx; x++) {
            double s = 0, w = 0;
            for (int i = 0; i < J->n; i++) {
                if (J->mk[i] && !J->mk[i][r + x]) continue;   /* the input has no shard here (occupancy air, outside its box): weight 0 */
                const double wi = (double)J->wz[(size_t)i * J->S + z] * J->wy[(size_t)i * J->S + y] * J->wx[(size_t)i * J->S + x];
                s += wi * J->in[i][r + x]; w += wi;
            }
            J->out[r + x] = w > 0 ? (uint8_t)fmin(255.0, s / w + 0.5) : 0;
        }
    }
}
/* one row segment [x0, x1) of block mode: inputs with p[i] = 0 skipped, as a 0 in their mask (ib_rows' arithmetic, same order) */
static void ib_seg(const ib_job *J, int z, int64_t y, int64_t x0, int64_t x1, const uint8_t *p) {
    const size_t r = ((size_t)z * J->n3[1] + y) * J->n3[2];
    if (J->n == 2) {
        volatile double wzy0v = (double)J->wz[z] * J->wy[y], wzy1v = (double)J->wz[(size_t)J->S + z] * J->wy[(size_t)J->S + y];
        const double wzy0 = wzy0v, wzy1 = wzy1v; const int u0 = p[0], u1 = p[1];
        const float *wx0 = J->wx, *wx1 = J->wx + J->S; const uint8_t *i0 = J->in[0] + r, *i1 = J->in[1] + r; uint8_t *o = J->out + r;
        for (int64_t x = x0; x < x1; x++) {
            double s = 0, w = 0;
            if (u0) { const double wi = wzy0 * wx0[x]; s += wi * i0[x]; w += wi; }
            if (u1) { const double wi = wzy1 * wx1[x]; s += wi * i1[x]; w += wi; }
            o[x] = w > 0 ? (uint8_t)fmin(255.0, s / w + 0.5) : 0;
        }
        return;
    }
    for (int64_t x = x0; x < x1; x++) {
        double s = 0, w = 0;
        for (int i = 0; i < J->n; i++) {
            if (!p[i]) continue;
            const double wi = (double)J->wz[(size_t)i * J->S + z] * J->wy[(size_t)i * J->S + y] * J->wx[(size_t)i * J->S + x];
            s += wi * J->in[i][r + x]; w += wi;
        }
        J->out[r + x] = w > 0 ? (uint8_t)fmin(255.0, s / w + 0.5) : 0;
    }
}
/* --tile-list FILE (2026-11-20, fullvram: the FAST ink score): only the output shards listed ("iz iy ix" of --shard) are blended and
   written; set by cmd_ink_blend for one ink_blend_run (mgpu: main) */
static const uint8_t *g_ib_keep; static int64_t g_ib_kn[3];
int cmd_ink_blend(int argc, char **argv) {
    const char *in[32]; int n = 0; const char *out = nullptr, *tile = nullptr, *grids = nullptr, *origins = nullptr;
    /* q 0 (lossless): the inputs are already q-coded by predict; a second lossy pass (the old default 32) stacked errors (urender audit 11-14) */
    double um = 0, q = 0; int nlev = 4, S = 512, nt = ufsm_threads(0, 16); const char *tlf = nullptr;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], opt_n("--um")) && i + 1 < argc) { um = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], opt_n("--tile")) && i + 1 < argc) { tile = argv[++i]; continue; }
        if (!strcmp(argv[i], opt_n("--grids")) && i + 1 < argc) { grids = argv[++i]; continue; }
        if (!strcmp(argv[i], opt_n("--origins")) && i + 1 < argc) { origins = argv[++i]; continue; }
        if (!strcmp(argv[i], opt_n("--levels")) && i + 1 < argc) { nlev = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], opt_n("--shard")) && i + 1 < argc) { S = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], opt_n("--q")) && i + 1 < argc) { q = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], opt_n("--threads")) && i + 1 < argc) { nt = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], opt_n("--tile-list")) && i + 1 < argc) { tlf = argv[++i]; continue; }
        if (!strncmp(argv[i], "--", 2)) { fprintf(stderr, "ink-blend: unknown option %s\n", argv[i]); return 2; }
        if (!out) out = argv[i]; else if (n < 32) in[n++] = argv[i]; else { fprintf(stderr, "ink-blend: at most 32 inputs\n"); return 2; }
    }
    double st[3], g[32][3];
    if (!out || n < 1 || um <= 0 || !tile || !grids || sscanf(tile, "%lf,%lf,%lf", &st[0], &st[1], &st[2]) != 3 || !(st[0] > 0 && st[1] > 0 && st[2] > 0) || S < 128 || S % 128) {
        fprintf(stderr, "usage: ufsm ink-blend <out> --um U --tile SZ,SY,SX --grids z,y,x:z,y,x:... <in1> <in2> ... [--origins z,y,x:...] [--levels 4] [--shard 512] [--q 0 (lossless; volcomp q)] [--threads 16] [--tile-list FILE (only these output shards)]\n"); return 2; }
    { int k = 0; char *t = strdup(grids);
      for (char *q = strtok(t, ":"); q; q = strtok(nullptr, ":")) { if (k < 32 && sscanf(q, "%lf,%lf,%lf", &g[k][0], &g[k][1], &g[k][2]) != 3) { free(t); fprintf(stderr, "ink-blend: --grids z,y,x:...\n"); return 2; } k++; }
      free(t); if (k != n) { fprintf(stderr, "ink-blend: %d grids for %d inputs\n", k, n); return 2; } }
    int64_t org[32][3] = {{0}};   /* --origins: output voxel = input i voxel + origin i (predicts of a --box without --full-grid); outside an input its weight is 0 */
    if (origins) { int k = 0; char *t = strdup(origins);
      for (char *q = strtok(t, ":"); q; q = strtok(nullptr, ":")) { long long a, b, c; if (k < 32 && sscanf(q, "%lld,%lld,%lld", &a, &b, &c) != 3) { free(t); fprintf(stderr, "ink-blend: --origins z,y,x:...\n"); return 2; } if (k < 32) { org[k][0] = a; org[k][1] = b; org[k][2] = c; } k++; }
      free(t); if (k != n) { fprintf(stderr, "ink-blend: %d origins for %d inputs\n", k, n); return 2; } }
    uint8_t *keep = nullptr;
    if (tlf) {   /* the listed shards on a grid big enough for any index in the file (indices < 0 ignored) */
        FILE *f = fopen(tlf, "r"); if (!f) { fprintf(stderr, "ink-blend: cannot open --tile-list %s\n", tlf); return 1; }
        long long a, b, c; int64_t mx[3] = {0, 0, 0}; long nl = 0;
        while (fscanf(f, "%lld %lld %lld", &a, &b, &c) == 3) { const long long t3[3] = {a, b, c}; for (int d = 0; d < 3; d++) if (t3[d] + 1 > mx[d]) mx[d] = t3[d] + 1; nl++; }
        if (!nl) { fclose(f); fprintf(stderr, "ink-blend: --tile-list %s is empty\n", tlf); return 1; }
        keep = calloc((size_t)(mx[0] * mx[1] * mx[2]), 1); rewind(f);
        while (keep && fscanf(f, "%lld %lld %lld", &a, &b, &c) == 3) if (a >= 0 && b >= 0 && c >= 0) keep[((size_t)a * mx[1] + b) * mx[2] + c] = 1;
        fclose(f); if (!keep) { fprintf(stderr, "ink-blend: out of memory\n"); return 1; }
        g_ib_keep = keep; for (int d = 0; d < 3; d++) g_ib_kn[d] = mx[d];
        fprintf(stderr, "ink-blend: --tile-list %s: %ld shards of %d\n", tlf, nl, S);
    }
    const int rc = ink_blend_run(out, um, st, n, in, g, org, nlev, S, q, nt, "{\"ufsm\":{\"content\":\"ink probability (ink-blend of overlapping predicts)\"}}");
    g_ib_keep = nullptr; free(keep); return rc;
}

/* ink_blend_run's shared state and its workers (perf 11-25): each worker takes the next output shard and reads, blends and writes it
   alone (its own buffers; z3_read and z3w_write_shard are thread-safe across shards; inner threads = --threads / workers) */
typedef struct {
    int n, S, B; z3 **zz; const char *const *in; const double *st; const double (*g)[3]; const int64_t (*org)[3]; z3w *w;
    int64_t ish[32][3], shd[32][3], pn[32][3]; uint8_t *pm[32];   /* per input: shape, shard edge, shard grid, presence map */
    const int64_t *job; long nj; int inner; double t0;
    atomic_long next, done; atomic_llong nlive, nblk; atomic_int err; char msg[512]; pthread_mutex_t mu;
} ib_ctx;
static int ib_present(const ib_ctx *C, int i, int64_t a, int64_t b, int64_t c) {
    if (a < 0 || b < 0 || c < 0 || a >= C->pn[i][0] || b >= C->pn[i][1] || c >= C->pn[i][2]) return 0;
    return C->pm[i][(size_t)((a * C->pn[i][1] + b) * C->pn[i][2] + c)];
}
static void ib_fail(ib_ctx *C, const char *m) {
    pthread_mutex_lock(&C->mu); if (!atomic_load(&C->err)) snprintf(C->msg, sizeof C->msg, "%s", m); atomic_store(&C->err, 1); pthread_mutex_unlock(&C->mu);
}
static void *ib_worker(void *arg) {
    ib_ctx *C = arg; const int n = C->n, S = C->S, B = C->B, nb = B ? S / B : 0, nb3 = nb * nb * nb; const size_t sv = (size_t)S * S * S;
    uint8_t *buf[32] = {nullptr}, *mk[32] = {nullptr}, *ob = malloc(sv), *bpres = B ? malloc((size_t)nb3 * n) : nullptr; float *wv[3]; int ok = ob && (!B || bpres);
    for (int d = 0; d < 3; d++) { wv[d] = malloc((size_t)n * S * sizeof(float)); ok &= wv[d] != nullptr; }
    for (int i = 0; i < n; i++) { buf[i] = malloc(sv); ok &= buf[i] != nullptr; }
    if (!ok) ib_fail(C, "ink-blend: out of memory");
    for (long j; ok && !atomic_load(&C->err) && (j = atomic_fetch_add(&C->next, 1)) < C->nj;) {
        const int64_t a = C->job[3 * j], b = C->job[3 * j + 1], c = C->job[3 * j + 2], o[3] = {a * S, b * S, c * S}, nn[3] = {S, S, S};
        long long live = 0; char em[512];
        for (int q = 0; q < nb3; q++) {   /* each block's inputs present (aligned: a block lies inside one input shard, present or absent whole) */
            const int bz = q / (nb * nb), by = q / nb % nb, bx = q % nb; int np = 0;
            for (int i = 0; i < n; i++) {
                int in = 1; int64_t q3[3];
                for (int d = 0; d < 3; d++) { const int64_t l = o[d] + (int64_t)(d ? d == 1 ? by : bx : bz) * B - C->org[i][d]; if (l < 0 || l >= C->ish[i][d]) in = 0; q3[d] = l >= 0 ? l / C->shd[i][d] : 0; }
                bpres[(size_t)q * n + i] = in && ib_present(C, i, q3[0], q3[1], q3[2]); np += bpres[(size_t)q * n + i];
            }
            live += np > 0;
        }
        for (int i = 0; i < n && ok; i++) {
            const int64_t lo[3] = {o[0] - C->org[i][0], o[1] - C->org[i][1], o[2] - C->org[i][2]};
            if (z3_read(C->zz[i], lo, nn, buf[i], C->inner)) { snprintf(em, sizeof em, "ink-blend: read %s: %s", C->in[i], z3_error()); ib_fail(C, em); ok = 0; break; }
            if (B) { free(mk[i]); mk[i] = nullptr; }   /* (block mode: the absent input shards are the blocks' presence flags, bpres) */
            else {   /* input shards absent inside this output shard (the other input's grid, occupancy-skipped air): a voxel mask of weight 0 */
                int part = 0;
                for (int pass = 0; pass < 2; pass++) {
                    for (int64_t za = lo[0] < 0 ? 0 : lo[0] / C->shd[i][0]; za * C->shd[i][0] < lo[0] + S && za * C->shd[i][0] < C->ish[i][0]; za++)
                        for (int64_t ya = lo[1] < 0 ? 0 : lo[1] / C->shd[i][1]; ya * C->shd[i][1] < lo[1] + S && ya * C->shd[i][1] < C->ish[i][1]; ya++)
                            for (int64_t xa = lo[2] < 0 ? 0 : lo[2] / C->shd[i][2]; xa * C->shd[i][2] < lo[2] + S && xa * C->shd[i][2] < C->ish[i][2]; xa++) {
                                if (ib_present(C, i, za, ya, xa)) continue;
                                if (!pass) { part = 1; continue; }
                                int64_t a0[3], a1[3]; const int64_t qa[3] = {za, ya, xa};
                                for (int d = 0; d < 3; d++) { a0[d] = qa[d] * C->shd[i][d] - lo[d]; a1[d] = a0[d] + C->shd[i][d]; if (a0[d] < 0) a0[d] = 0; if (a1[d] > S) a1[d] = S; }
                                for (int64_t z = a0[0]; z < a1[0]; z++) for (int64_t y = a0[1]; y < a1[1]; y++) memset(mk[i] + ((size_t)z * S + y) * S + a0[2], 0, (size_t)(a1[2] - a0[2]));
                            }
                    if (!pass) { if (!part) break; if (!mk[i] && !(mk[i] = malloc(sv))) { ib_fail(C, "ink-blend: out of memory"); ok = 0; break; } memset(mk[i], 1, sv); }
                }
                if (!part && mk[i]) { free(mk[i]); mk[i] = nullptr; }
            }
            for (int d = 0; d < 3; d++) for (int k = 0; k < S; k++)   /* outside input i: weight 0 */
                wv[d][(size_t)i * S + k] = lo[d] + k < 0 || lo[d] + k >= C->ish[i][d] ? 0.f : (float)fb_tile_w(o[d], 1, k, C->st[d], C->g[i][d]);
        }
        if (!ok) break;
        ib_job J = {buf, n, wv[0], wv[1], wv[2], S, {S, S, S}, ob, mk, B ? bpres : nullptr, B, nb};
        if (C->inner > 1) fit_par(S, ib_rows, &J); else ib_rows(&J, 0, S, 0);
        atomic_fetch_add(&C->nlive, live); atomic_fetch_add(&C->nblk, nb3);
        if (z3w_write_shard(C->w, a, b, c, ob, C->inner)) { snprintf(em, sizeof em, "%s", z3w_error()); ib_fail(C, em); break; }
        const long k = atomic_fetch_add(&C->done, 1) + 1;
        if (k % 50 == 0) fprintf(stderr, "ink-blend: %ld of %ld shards %.0f s\n", k, C->nj, now() - C->t0);
    }
    free(ob); free(bpres); for (int d = 0; d < 3; d++) free(wv[d]); for (int i = 0; i < n; i++) { free(buf[i]); free(mk[i]); }
    return nullptr;
}

/* the blend itself (ink-blend; predict --seamless): n uint8 probability groups (level named by um) on overlapping tile grids (tile st,
   grid offsets g, level voxels; input voxel + org = output voxel) -> out/<um> + nlev - 1 mean-pooled levels, shards of S, volcomp q */
int ink_blend_run(const char *out, double um, const double st[3], int n, const char *const *in, const double (*g)[3], const int64_t (*org)[3],
                  int nlev, int S, double q, int nt, const char *attrs) {
    char lv[32]; z3w_level_name(um, lv, sizeof lv);
    store *ss[32]; z3 *zz[32]; int64_t shape[3] = {0, 0, 0}, ish[32][3];
    for (int i = 0; i < n; i++) {
        ss[i] = store_open(in[i]); zz[i] = ss[i] ? z3_open(ss[i], lv, nullptr) : nullptr;
        if (!zz[i]) { fprintf(stderr, "ink-blend: cannot open %s/%s: %s\n", in[i], lv, z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(zz[i]);
        for (int d = 0; d < 3; d++) { ish[i][d] = m->shape[d]; if (org[i][d] + m->shape[d] > shape[d]) shape[d] = org[i][d] + m->shape[d]; }   /* the output covers every input */
    }
    char odir[1400]; snprintf(odir, sizeof odir, "%s/%s", out, lv);
    z3w *w = z3w_create(odir, shape, S, (float)q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    int64_t ns[3]; for (int d = 0; d < 3; d++) ns[d] = (shape[d] + S - 1) / S;
    /* block flags instead of voxel masks (perf 11-25): predict --seamless 2 writes its passes with shards S / 2 (128 for window 288 halo
       16) and blends into the run's --shard (512): a tile-list predict fills a few 128^3 blocks of each 512^3 output shard, and every
       input with an absent shard got a 128 MB voxel mask (memset to 1, then 0 per absent shard) and every voxel was blended (Paris4 aux,
       1455 tiles: the blend 68 of 189 s wall, memset 18 % + ib_rows 18 % of its CPU). With every input's shards and origin on a B grid
       (B = the smallest input shard edge, dividing S) each B^3 block lies inside one shard per input: a per-block presence flag replaces
       the mask, blocks no input has are written 0 (the full pass gives them 0 too: every weight masked), the rest blended with the same
       weights, values and order (byte-identical; tests/test_ink_blend_block.c). (A per-block read + blend was slower: 105 s; the
       whole-shard reads stay.) UFSM_IB_BLOCK=0: the voxel masks. */
    int B = S;
    for (int i = 0; i < n; i++) { const z3_meta *m = z3_meta_of(zz[i]); for (int d = 0; d < 3; d++) if (m->shard[d] < B) B = (int)m->shard[d]; }
    { int ok = B >= 16 && B < S && S % B == 0 && !(getenv("UFSM_IB_BLOCK") && !atoi(getenv("UFSM_IB_BLOCK")));
      for (int i = 0; i < n && ok; i++) { const z3_meta *m = z3_meta_of(zz[i]); for (int d = 0; d < 3; d++) if (m->shard[d] % B || org[i][d] % B) ok = 0; }
      if (!ok) B = 0; }
    /* shard presence once per input (perf 11-25, inkperf R11d HOLD: a seamless tile-list blend walked ~70 k output shards asking
       z3_shard_present per input shard, an index open each, > 82 s on one thread with the GPU idle): z3_shard_map lists a local store's
       shard files, then the output shards with any input are blended by up to --threads workers, one shard each (same bytes: the
       per-voxel arithmetic does not depend on the split) */
    ib_ctx C = {.n = n, .S = S, .B = B, .zz = zz, .in = in, .st = st, .g = g, .org = org, .w = w, .mu = PTHREAD_MUTEX_INITIALIZER};
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) { C.ish[i][d] = ish[i][d]; C.shd[i][d] = z3_meta_of(zz[i])->shard[d]; }
        if (z3_shard_map(zz[i], C.pn[i], &C.pm[i]) < 0) { fprintf(stderr, "ink-blend: shard map of %s: %s\n", in[i], z3_error()); return 1; }
    }
    int64_t *job = malloc((size_t)(ns[0] * ns[1] * ns[2] > 0 ? ns[0] * ns[1] * ns[2] : 1) * 3 * sizeof *job); long nj = 0; int nskip = 0;
    if (!job) { fprintf(stderr, "ink-blend: out of memory\n"); return 1; }
    for (int64_t a = 0; a < ns[0]; a++) for (int64_t b = 0; b < ns[1]; b++) for (int64_t c = 0; c < ns[2]; c++) {
        if (g_ib_keep && (a >= g_ib_kn[0] || b >= g_ib_kn[1] || c >= g_ib_kn[2] || !g_ib_keep[((size_t)a * g_ib_kn[1] + b) * g_ib_kn[2] + c])) { nskip++; continue; }
        const int64_t o[3] = {a * S, b * S, c * S}; int any = 0;
        for (int i = 0; i < n && !any; i++) {   /* any input shard holding this output shard's box (input-local voxels o - origin, clipped) */
            int64_t lo[3], hi[3]; int ok = 1;
            for (int d = 0; d < 3; d++) { lo[d] = o[d] - org[i][d]; hi[d] = lo[d] + S - 1; if (lo[d] < 0) lo[d] = 0; if (hi[d] > ish[i][d] - 1) hi[d] = ish[i][d] - 1; if (lo[d] > hi[d]) ok = 0; }
            if (!ok) continue;
            for (int64_t za = lo[0] / C.shd[i][0]; za <= hi[0] / C.shd[i][0] && !any; za++)
                for (int64_t ya = lo[1] / C.shd[i][1]; ya <= hi[1] / C.shd[i][1] && !any; ya++)
                    for (int64_t xa = lo[2] / C.shd[i][2]; xa <= hi[2] / C.shd[i][2] && !any; xa++) any = ib_present(&C, i, za, ya, xa);
        }
        if (!any) { nskip++; continue; }
        job[3 * nj] = a; job[3 * nj + 1] = b; job[3 * nj + 2] = c; nj++;
    }
    int nwk = nt < 1 ? 1 : nt > 64 ? 64 : nt; if (nwk > nj) nwk = nj > 0 ? (int)nj : 1;
    C.job = job; C.nj = nj; C.inner = nt / nwk > 1 ? nt / nwk : 1; C.t0 = now();
    pthread_t th[64]; int nth = 0;
    for (int k = 1; k < nwk; k++) if (!pthread_create(&th[nth], nullptr, ib_worker, &C)) nth++;
    ib_worker(&C);
    for (int k = 0; k < nth; k++) pthread_join(th[k], nullptr);
    for (int i = 0; i < n; i++) free(C.pm[i]);
    free(job);
    if (atomic_load(&C.err)) { fprintf(stderr, "%s\n", C.msg); return 1; }
    z3w_close(w);
    fprintf(stderr, "ink-blend: %ld shards written, %d empty, %.0f s (%d workers)", nj, nskip, now() - C.t0, nth + 1);
    if (B) fprintf(stderr, " (blocks of %d: %lld of %lld blended)", B, (long long)atomic_load(&C.nlive), (long long)atomic_load(&C.nblk));
    fputc('\n', stderr);
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um, l, shape, S, (float)q, 0, nt, attrs)) return 1;
    pyramid_write_group(out, um, nlev, "ink-blend", attrs);
    for (int i = 0; i < n; i++) { z3_close(zz[i]); store_close(ss[i]); }
    return 0;
}
