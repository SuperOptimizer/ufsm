/* ufsm predict: sliding-window inference of a checkpoint over a CT box, written as a volcomp zarr v3
   probability pyramid (uint8 p*255, q=8) in the published layout. */
#include "nn.h"
#include "sysinfo.h"
#include "checkpoint.h"
#include "sheet.h"
#include <zstd.h>
#include <sys/stat.h>
#include "papyrus.h"
#include "scan_augment.h"
#include "pyramid.h"
#include "sources.h"
#include "split.h"
#include "store.h"
#include "unet.h"
#include "fit.h"
#include "tta.h"
#include "view.h"
#include "sweep.h"
#include "z3w.h"
#include "zarr3.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void pred_axis(unet *u, int valid, int D, const float *dy, const float *dx);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* tile read-ahead: a reader thread fills a ring of CT windows (with their nonzero count and intensity sums) in tile
   order while the main thread runs the network on the previous one */
#define NSLOT 3
typedef struct { int64_t tz, ty, tx; int64_t shard; int noout; } tile_t;   /* noout: --feather, computed for its spills only (written already / another worker's) */
typedef struct {
    z3 *ct; const int64_t *bo; int W, Wz, nthreads;   /* windows Wz x W x W */
    tile_t *tiles; long ntiles;
    uint8_t *buf[NSLOT]; size_t nz[NSLOT]; double sum[NSLOT], sq[NSLOT];
    atomic_long filled;          /* tiles read so far (slot = i % NSLOT) */
    atomic_long consumed;        /* tiles released by the main thread */
    atomic_int failed;
    pthread_mutex_t mu; pthread_cond_t cv;
    double t_read, t_stats, t_wait;   /* UFSM_PRED_PROF: seconds in z3_read, the stats scan, waiting for a free slot */
    int nostats;                      /* the window statistics are computed on the device (default path) */
} reader_t;
static void *reader_main(void *arg) {
    reader_t *r = arg;
    size_t w3 = (size_t)r->Wz * r->W * r->W;
    for (long i = 0; i < r->ntiles && !atomic_load(&r->failed); i++) {
        double tw = now();
        pthread_mutex_lock(&r->mu);
        while (i - atomic_load(&r->consumed) >= NSLOT) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        double t1 = now(); r->t_wait += t1 - tw;
        int k = (int)(i % NSLOT);
        tile_t *t = &r->tiles[i];
        int64_t o[3] = {r->bo[0] + t->tz, r->bo[1] + t->ty, r->bo[2] + t->tx}, n[3] = {r->Wz, r->W, r->W};
        if (z3_read(r->ct, o, n, r->buf[k], r->nthreads)) { atomic_store(&r->failed, 1); }
        else if (r->nostats) r->t_read += now() - t1;
        else {
            double t2 = now(); r->t_read += t2 - t1;
            const uint8_t *c = r->buf[k]; size_t nz = 0; double sum = 0, sq = 0;
            for (size_t j = 0; j < w3; j++) { nz += c[j] != 0; sum += c[j]; sq += (double)c[j] * c[j]; }
            r->nz[k] = nz; r->sum[k] = sum; r->sq[k] = sq;
            r->t_stats += now() - t2;
        }
        pthread_mutex_lock(&r->mu); atomic_store(&r->filled, i + 1); pthread_cond_broadcast(&r->cv); pthread_mutex_unlock(&r->mu);
    }
    pthread_mutex_lock(&r->mu); pthread_cond_broadcast(&r->cv); pthread_mutex_unlock(&r->mu);
    return nullptr;
}
/* shard writer: the main thread hands a filled shard buffer to a writer thread and continues into the next buffer of a ring of nb
   (2 by default; one window per k^3 shards: up to 2 k^3, UFSM_PRED_WBUF, so the shards of a window are compressed while the next window
   runs instead of the main thread waiting on each shard's predecessor); buffers are written in submission (ring) order */
#define WBUF_MAX 32
typedef struct { z3w *w; int nthreads; size_t bytes; uint8_t *buf[WBUF_MAX]; int64_t sz[WBUF_MAX], sy[WBUF_MAX], sx[WBUF_MAX]; int pending[WBUF_MAX]; int nb, next;
                 atomic_int failed; pthread_mutex_t mu; pthread_cond_t cv; int stop;
                 int gpu, async, skip; void *evs[WBUF_MAX], *evd[WBUF_MAX]; double t_write; long n_write;
                 int64_t off[3]; int donefd; } writer_t;   /* off: shard index offset (--full-grid); donefd: the --resume log (-1 none) */   /* (UFSM_PRED_PROF: seconds compressing / writing, shards) */   /* async: buffer k arrives by a copy on the offload stream (evd[k]) */
static void *writer_main(void *arg) {
    writer_t *W = arg;
    if (W->async) nn_init(W->gpu);
    for (;;) {
        pthread_mutex_lock(&W->mu);
        int k = -1;
        while (k < 0 && !W->stop) { if (W->pending[W->next]) k = W->next; else pthread_cond_wait(&W->cv, &W->mu); }
        if (k < 0) { pthread_mutex_unlock(&W->mu); return nullptr; }
        pthread_mutex_unlock(&W->mu);
        if (W->async) nn_event_sync(W->evd[k]);   /* the shard's device-to-host copy (queued without blocking the main thread) */
        const double tw = now();
        if (!W->skip && z3w_write_shard(W->w, W->sz[k] + W->off[0], W->sy[k] + W->off[1], W->sx[k] + W->off[2], W->buf[k], W->nthreads)) atomic_store(&W->failed, 1);   /* skip: --no-out 1 */
        else if (W->donefd >= 0) dprintf(W->donefd, "%lld %lld %lld\n", (long long)(W->sz[k] + W->off[0]), (long long)(W->sy[k] + W->off[1]), (long long)(W->sx[k] + W->off[2]));   /* (O_APPEND: whole lines across GPU workers) */
        W->t_write += now() - tw; W->n_write++;
        pthread_mutex_lock(&W->mu); W->pending[k] = 0; W->next = (k + 1) % W->nb; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
}
/* the device shard dsh into buffer k: on the offload stream (the main stream waits for it on the device before dsh is reused, the writer
   thread before it compresses), so the main thread does not block on the copy; else a plain copy */
static void writer_d2h(writer_t *W, int k, const uint8_t *dsh) {
    if (!W->async) { nn_d2h(W->buf[k], dsh, W->bytes); return; }
    nn_offload_copy(W->buf[k], dsh, W->bytes, 1, W->evs[k], W->evd[k]);
    nn_stream_wait(0, W->evd[k]);
}
/* queue buffer k for writing; returns when the next buffer of the ring (writer_next) is free to fill */
static int writer_next(const writer_t *W, int k) { return (k + 1) % W->nb; }
static int writer_submit(writer_t *W, int k, int64_t sz, int64_t sy, int64_t sx) {
    pthread_mutex_lock(&W->mu);
    W->sz[k] = sz; W->sy[k] = sy; W->sx[k] = sx; W->pending[k] = 1; pthread_cond_broadcast(&W->cv);
    while (W->pending[writer_next(W, k)]) pthread_cond_wait(&W->cv, &W->mu);
    pthread_mutex_unlock(&W->mu);
    return atomic_load(&W->failed);
}
/* predict --feather B: seam-free single pass. Windows on the shard grid (interior S = k shards) in z-y-x order; each window's prediction
   E over [I, I + S + B) per axis (its interior plus B voxels of its upper halo; B <= halo) is downloaded. The output over the interior
   is E itself except in the lower shell (any coordinate < B), where the earlier windows' upper spills are blended in: weight of the
   window shifted back by delta (delta_d in {0,1}) = prod_d (delta_d ? 1 - u(v_d) : u(v_d)), u(j) = sin^2(pi/2 (j + .5) / B) for j < B
   (else 1 for delta_d = 0, 0 for delta_d = 1), normalized over the windows present (air / box edges). The transition is a smooth ramp
   of B voxels just above every window border; only one-sided spills of earlier windows are held (an upper z slab per window of the
   current and previous window layer, an upper y slab per window of this and the previous row, the previous window's upper x slab). */
typedef struct { int64_t id; uint8_t *p; } fe_slot;
typedef struct {
    int S, B, E;                  /* interior, band, S + B */
    int64_t nss[3];               /* windows per axis */
    fe_slot *zs, *ys, xs;         /* zs[2][ny*nx] (layer parity), ys[2][nx] (row parity) */
    const uint8_t *e;             /* this window's E (E^3, pinned) */
    const uint8_t *nb[8];         /* the spill source of the window shifted back by delta (bit 0 x, 1 y, 2 z), nullptr = absent */
    float u[2][1024];             /* [0] own weight u(j), [1] = 1 - u(j), j < B */
    int64_t n[3];                 /* the interior clipped to the box */
    uint8_t *out; int64_t oo[3], on[3]; int osh;   /* output: shard buffer (osh^3) at interior offset oo, extent on */
} feather_t;
static uint8_t fe_val(const feather_t *F, int d, int z, int y, int x) {   /* the value of window (back by d) at this window's interior voxel */
    const int S = F->S, E = F->E, B = F->B;
    const uint8_t *p = F->nb[d];
    if (!d) return F->e[((size_t)z * E + y) * E + x];
    const int uz = z + (d & 4 ? S : 0), uy = y + (d & 2 ? S : 0), ux = x + (d & 1 ? S : 0);
    if (uz >= S) return p[((size_t)(uz - S) * E + uy) * E + ux];          /* its z slab: B x E x E */
    if (uy >= S) return p[((size_t)uz * B + (uy - S)) * E + ux];          /* its y slab: S x B x E */
    return p[((size_t)uz * S + uy) * B + (ux - S)];                       /* its x slab: S x S x B */
}
static uint8_t fe_blend(const feather_t *F, int z, int y, int x) {   /* a lower-shell voxel: the normalized blend over the windows present */
    const int B = F->B; float s = 0, w = 0;
    for (int d = 0; d < 8; d++) {
        if (d && !F->nb[d]) continue;
        const int dz = d >> 2 & 1, dy = d >> 1 & 1, dx = d & 1;
        if ((dz && z >= B) || (dy && y >= B) || (dx && x >= B)) continue;
        const float wi = (z < B ? F->u[dz][z] : 1.f) * (y < B ? F->u[dy][y] : 1.f) * (x < B ? F->u[dx][x] : 1.f);
        s += wi * fe_val(F, d, z, y, x); w += wi;
    }
    return w > 0 ? (uint8_t)fminf(255.f, s / w + 0.5f) : 0;
}
static void fe_rows(void *a, int r0, int r1, int t) {   /* rows r = lz * on[1] + ly (the lower-shell rows spread over every thread) */
    (void)t; const feather_t *F = a; const int B = F->B, E = F->E, sh = F->osh;
    const int64_t nr = F->on[0] * F->on[1];
    for (int q = r0; q < r1; q++) {   /* (q -> r: times the prime 2^31 - 1 > nr, a bijection, so each thread's range mixes shell and core rows) */
        const int64_t r = (int64_t)((uint64_t)q * 2147483647u % (uint64_t)nr), lz = r / F->on[1], ly = r % F->on[1];
        const int z = (int)(F->oo[0] + lz), y = (int)(F->oo[1] + ly), x0 = (int)F->oo[2], nx = (int)F->on[2];
        uint8_t *o = F->out + ((size_t)lz * sh + ly) * sh;
        const uint8_t *er = F->e + ((size_t)z * E + y) * E + x0;
        if (z < B || y < B) { for (int lx = 0; lx < nx; lx++) o[lx] = fe_blend(F, z, y, x0 + lx); continue; }
        int lx = 0;
        for (; lx < nx && x0 + lx < B; lx++) o[lx] = fe_blend(F, z, y, x0 + lx);   /* (the x shell), the rest this window's own */
        if (lx < nx) memcpy(o + lx, er + lx, (size_t)(nx - lx));
    }
}
static int feather_init(feather_t *F, int S, int B, const int64_t nss[3]) {
    memset(F, 0, sizeof *F); F->S = S; F->B = B; F->E = S + B;
    for (int d = 0; d < 3; d++) F->nss[d] = nss[d];
    for (int j = 0; j < B; j++) { const double s = sin(M_PI / 2 * (j + 0.5) / B); F->u[0][j] = (float)(s * s); F->u[1][j] = (float)(1 - s * s); }
    F->zs = calloc((size_t)(2 * nss[1] * nss[2] + 2 * nss[2]), sizeof *F->zs); F->ys = F->zs + 2 * nss[1] * nss[2];
    if (!F->zs) return 1;
    for (int64_t k = 0; k < 2 * nss[1] * nss[2] + 2 * nss[2]; k++) F->zs[k].id = -1;
    F->xs.id = -1; return 0;
}
static void feather_free(feather_t *F) {
    if (!F->zs) return;
    for (int64_t k = 0; k < 2 * F->nss[1] * F->nss[2] + 2 * F->nss[2]; k++) free(F->zs[k].p);
    free(F->xs.p); free(F->zs); F->zs = nullptr;
}
/* the spill sources of window (a, b, c) (window grid), before it is blended */
static void feather_lookup(feather_t *F, int64_t a, int64_t b, int64_t c) {
    const int64_t ny = F->nss[1], nx = F->nss[2];
    for (int d = 1; d < 8; d++) {
        const int64_t qa = a - (d >> 2 & 1), qb = b - (d >> 1 & 1), qc = c - (d & 1);
        F->nb[d] = nullptr;
        if (qa < 0 || qb < 0 || qc < 0) continue;
        const int64_t id = (qa * ny + qb) * nx + qc; const fe_slot *s;
        if (d & 4) s = &F->zs[(qa & 1) * ny * nx + qb * nx + qc];
        else if (d & 2) s = &F->ys[(qb & 1) * nx + qc];
        else s = &F->xs;
        if (s->id == id) F->nb[d] = s->p;
    }
}
/* keep window (a, b, c)'s upper slabs of E for the windows after it; 1 on out of memory */
static int feather_keep(feather_t *F, int64_t a, int64_t b, int64_t c) {
    const int64_t ny = F->nss[1], nx = F->nss[2], id = (a * ny + b) * nx + c; const int S = F->S, B = F->B, E = F->E;
    fe_slot *sz = &F->zs[(a & 1) * ny * nx + b * nx + c], *sy = &F->ys[(b & 1) * nx + c], *sx = &F->xs;
    if (!sz->p && !(sz->p = malloc((size_t)B * E * E))) return 1;
    if (!sy->p && !(sy->p = malloc((size_t)S * B * E))) return 1;
    if (!sx->p && !(sx->p = malloc((size_t)S * S * B))) return 1;
    memcpy(sz->p, F->e + (size_t)S * E * E, (size_t)B * E * E); sz->id = id;
    for (int z = 0; z < S; z++) memcpy(sy->p + (size_t)z * B * E, F->e + ((size_t)z * E + S) * E, (size_t)B * E);
    sy->id = id;
    for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) memcpy(sx->p + ((size_t)z * S + y) * B, F->e + ((size_t)z * E + y) * E + S, (size_t)B);
    sx->id = id;
    return 0;
}
/* the blend + shard hand-off of --feather windows on a thread of its own (windows in order; two pinned E buffers), so the GPU runs the
   next window meanwhile (the blend was 0.34 s per 576^3 window on the laptop, more than the forward) */
typedef struct { int k; int64_t a, b, c; int noout; } fe_job;
typedef struct {
    feather_t *F; writer_t *WR; int sb; int kss, shard; const int64_t *ns, *bn;
    uint8_t *fh[2]; int busy[2];
    fe_job q[2]; int qh, qn, stop, failed;
    pthread_mutex_t mu; pthread_cond_t cv; pthread_t th;
} fe_worker;
static void *fe_main(void *arg) {
    fe_worker *W = arg; feather_t *F = W->F; const int kss = W->kss, shard = W->shard;
    for (;;) {
        pthread_mutex_lock(&W->mu);
        while (!W->qn && !W->stop) pthread_cond_wait(&W->cv, &W->mu);
        if (!W->qn) { pthread_mutex_unlock(&W->mu); return nullptr; }
        const fe_job j = W->q[W->qh];
        pthread_mutex_unlock(&W->mu);
        int bad = 0;
        F->e = W->fh[j.k]; feather_lookup(F, j.a, j.b, j.c);
        if (!j.noout) for (int64_t qz = j.a * kss; qz < j.a * kss + kss && qz < W->ns[0] && !bad; qz++) for (int64_t qy = j.b * kss; qy < j.b * kss + kss && qy < W->ns[1] && !bad; qy++)
        for (int64_t qx = j.c * kss; qx < j.c * kss + kss && qx < W->ns[2] && !bad; qx++) {
            const int64_t q[3] = {qz, qy, qx}, Q[3] = {j.a, j.b, j.c}; int part_sh = 0;
            for (int d = 0; d < 3; d++) {
                F->oo[d] = (q[d] - Q[d] * kss) * shard;
                F->on[d] = W->bn[d] - q[d] * shard < shard ? W->bn[d] - q[d] * shard : shard;
                part_sh |= F->on[d] < shard;
            }
            uint8_t *sbuf = W->WR->buf[W->sb];
            if (part_sh) memset(sbuf, 0, W->WR->bytes);
            F->out = sbuf; F->osh = shard;
            fit_par((int)(F->on[0] * F->on[1]), fe_rows, F);
            if (writer_submit(W->WR, W->sb, qz, qy, qx)) bad = 1;
            W->sb = writer_next(W->WR, W->sb);
        }
        if (!bad && feather_keep(F, j.a, j.b, j.c)) bad = 2;
        pthread_mutex_lock(&W->mu);
        if (bad) W->failed = bad;
        W->busy[j.k] = 0; W->qh = (W->qh + 1) % 2; W->qn--;
        pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
}
static int fe_wait_idle(fe_worker *W) {   /* every queued window blended and handed to the writer; the failure code */
    pthread_mutex_lock(&W->mu);
    while (W->qn) pthread_cond_wait(&W->cv, &W->mu);
    const int f = W->failed; pthread_mutex_unlock(&W->mu);
    return f;
}
/* fit-field accumulators (12 planes, src/nn_fit.cu nn_fit_accum) of nf cells -> the 10 uint8 fit-field arrays */
static void fit_finalize(const float *acc, size_t nf, int h, uint8_t *const a[10]) {
    for (size_t i = 0; i < nf; i++) {
        const float cnt = acc[i], mat = acc[nf + i];
        a[0][i] = a[1][i] = 128; for (int k = 2; k < 10; k++) a[k][i] = 0;
        if (cnt <= 0 || mat <= 0) continue;
        a[9][i] = (uint8_t)lroundf(255.f * fminf(1.f, mat / cnt));
        a[0][i] = (uint8_t)lroundf(128.f + 127.f * acc[2 * nf + i] / mat); a[1][i] = (uint8_t)lroundf(128.f + 127.f * acc[3 * nf + i] / mat);
        for (int k = 0; k < 6; k++) a[2 + k][i] = (uint8_t)lroundf(255.f * fminf(1.f, fmaxf(0.f, (k < 3 ? 0.f : 0.5f) + acc[(4 + k) * nf + i] / mat)));
        const float rho = acc[10 * nf + i] / mat;   /* turns per native voxel */
        if (rho > 0) { const float v = 8.f / (h * rho); a[8][i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lroundf(v)); }
    }
}
/* --fit-zarr planes: task unwrap 11 (pc .. w, then the recto plane r), task unwrap2 14 (pc .. w, then the winding mod 4 planes dz dy
   dx dq: no recto head); with --fit-ct the CT planes ct and cs after them (the fitter loads planes by name) */
enum { FFZ_MAXP = 16 };
static const char *ffz_names[FFZ_MAXP] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "r"};
static int ffz_np = 11;   /* the planes written */
static int ffz_nb = 11;   /* the planes before ct / cs */
static const int ffz_fill[FFZ_MAXP] = {128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
/* --fit-zarr: write the slab buffer's field shards (z index fsz) of every array */
/* host field slab slab (FZ cells in z from slab FZ, FZ a multiple of the field shard FS; planes past F[0] not held) -> its rows of
   shards, the shards compressed and written on threads (a 128^3 field shard is one chunk, so z3w_write_shard alone is one thread:
   the 10 arrays of a 256 x 2048 x 2048 box took 163 s) */
/* the slab's layout: y cells in NR ring slots of YR cells (slot (y / YR) % NR), a slot HZ z cells x YR x F[2]; NR 1, YR F[1]: the whole
   cross-section. With row flushes and FZ = the (super-)shard row height, 3 slots of one (super-)shard row: a slot is taken by the row
   whose chunks arrive once its previous row is written and reset (own: the slab row held, -1 free), so the slab holds 3 rows instead of
   the cross-section (PHerc Paris 3 at 9.6 um, window 896: 108 -> 23 GB host, the bound on concurrent jobs) */
typedef struct { int HZ, YR, NR; int64_t nRR, *own; pthread_mutex_t *mu; pthread_cond_t *cv; } ffring;
static inline uint8_t *ffz_row(uint8_t *s, const ffring *R, int F2, int zin, int64_t y) {
    return s + (((size_t)((y / R->YR) % R->NR) * R->HZ + (size_t)zin) * R->YR + (size_t)(y % R->YR)) * F2;
}
static int ffring_owned(const ffring *R, int64_t slab, int64_t rr) {   /* the slab's ring row rr holds chunks (ring off: always) */
    if (R->NR == 1) return 1;
    pthread_mutex_lock(R->mu); const int o = R->own[rr % R->NR] == slab * R->nRR + rr; pthread_mutex_unlock(R->mu);
    return o;
}
static void ffring_release(const ffring *R, uint8_t *const sl[FFZ_MAXP], int64_t slab, int64_t rr, int F2) {   /* a written row: fill, slot free */
    const int s = (int)(rr % R->NR); const size_t n = (size_t)R->HZ * R->YR * F2;
    if (!ffring_owned(R, slab, rr)) return;
    for (int k = 0; k < ffz_np; k++) memset(sl[k] + (size_t)s * n, ffz_fill[k], n);
    pthread_mutex_lock(R->mu); R->own[s] = -1; pthread_cond_broadcast(R->cv); pthread_mutex_unlock(R->mu);
}
typedef struct { z3w *const *zw; uint8_t *const *sl; int64_t slab; const int *F; int FS, FZ, nrow, nx; const int *rq, *ry; const ffring *rg; _Atomic int next, fail; } ffz_job;
static void *ffz_worker(void *a) {   /* jobs: (array k, field-shard row (q, fsy) of the list, fsx) */
    ffz_job *J = a; const int FS = J->FS, *F = J->F; uint8_t *sb = malloc((size_t)FS * FS * FS);
    const int njob = ffz_np * J->nrow * J->nx;
    for (;;) {
        const int t = atomic_fetch_add(&J->next, 1); if (t >= njob || atomic_load(&J->fail)) break;
        const int k = t / (J->nrow * J->nx), r = t / J->nx % J->nrow, fsx = t % J->nx, q = J->rq[r], fsy = J->ry[r];
        const int64_t fsz = J->slab * (J->FZ / FS) + q;
        memset(sb, ffz_fill[k], (size_t)FS * FS * FS);
        for (int a2 = 0; a2 < FS && fsz * FS + a2 < F[0]; a2++) for (int b = 0; b < FS && (int64_t)fsy * FS + b < F[1]; b++) {
            const int64_t x0 = (int64_t)fsx * FS, nx = F[2] - x0 < FS ? F[2] - x0 : FS;
            memcpy(sb + ((size_t)a2 * FS + b) * FS, ffz_row(J->sl[k], J->rg, F[2], q * FS + a2, (int64_t)fsy * FS + b) + x0, (size_t)nx);
        }
        if (z3w_write_shard(J->zw[k], fsz, fsy, fsx, sb, 1)) atomic_store(&J->fail, 1);
    }
    free(sb); return nullptr;
}
/* field-shard rows (q, fsy) of the slab in [q0, q1) x [y0, y1) not yet written (done: [FZ / FS][ny] per slab, marked here); reset:
   the slab is left (its buffers back to the fill values, done cleared) */
static int ffz_flush(z3w *const zw[FFZ_MAXP], uint8_t *const sl[FFZ_MAXP], int64_t slab, const int F[3], int FS, int FZ, int q0, int q1, int y0, int y1, uint8_t *done, int reset,
                     int ntmax, const ffring *rg) {   /* ntmax: compression threads (0: every CPU) */
    const int ny = (F[1] + FS - 1) / FS;
    int nq = 0; for (int q = 0; q < FZ / FS; q++) if ((slab * (FZ / FS) + q) * FS < F[0]) nq = q + 1;
    if (q1 > nq) q1 = nq;
    if (y1 > ny) y1 = ny;
    int *rq = malloc(sizeof(int) * (size_t)(FZ / FS) * ny * 2), *ry = rq + (size_t)(FZ / FS) * ny, nrow = 0;
    if (!rq) return -1;
    for (int q = q0; q < q1; q++) for (int y = y0; y < y1; y++) if (!done[(size_t)q * ny + y]) {
        done[(size_t)q * ny + y] = 1;
        if (!ffring_owned(rg, slab, (int64_t)y * FS / rg->YR)) continue;   /* a ring row without chunks: fill, nothing to write */
        rq[nrow] = q; ry[nrow] = y; nrow++;
    }
    ffz_job J = {.zw = zw, .sl = sl, .slab = slab, .F = F, .FS = FS, .FZ = FZ, .nrow = nrow, .nx = (F[2] + FS - 1) / FS, .rq = rq, .ry = ry, .rg = rg};
    atomic_store(&J.next, 0); atomic_store(&J.fail, 0);
    long nt = ufsm_ncpu(); if (ntmax > 0 && nt > ntmax) nt = ntmax; if (nt > 64) nt = 64; if (nt < 1) nt = 1;
    pthread_t th[64]; int up[64] = {0};
    for (int i = 1; i < nt; i++) up[i] = pthread_create(&th[i], nullptr, ffz_worker, &J) == 0;
    ffz_worker(&J);
    for (int i = 1; i < nt; i++) if (up[i]) pthread_join(th[i], nullptr);
    free(rq);
    if (atomic_load(&J.fail)) return -1;
    if (!reset) return 0;
    for (int k = 0; k < ffz_np; k++) memset(sl[k], ffz_fill[k], (size_t)rg->NR * rg->HZ * rg->YR * F[2]);
    memset(done, 0, (size_t)(FZ / FS) * ny);
    if (rg->NR > 1) { pthread_mutex_lock(rg->mu); for (int s = 0; s < rg->NR; s++) rg->own[s] = -1; pthread_cond_broadcast(rg->cv); pthread_mutex_unlock(rg->mu); }
    return 0;
}

/* --fit-zarr host side off the main thread: per window chunk the device fields (11 B per cell, 1.5 GB for a 512^3 interior at
   --fit-h 1) go to pinned memory on the offload stream; one thread waits for each copy, scatters it into the slab buffers and
   flushes slabs in queue order (both used to run on the main thread while the GPU idled). Two staging buffers. */
typedef struct { int kind, b, c1, c2, nz, ny, nx; int64_t cz, slab; size_t nl; int q0, q1, y0, y1; } ffw_job;   /* kind 0: chunk copy, 1: slab flush (the rows
                                                                                                                   left, then reset), 2: rows q0..q1 x y0..y1 */
#define FFW_Q 256
typedef struct {
    pthread_mutex_t mu; pthread_cond_t cv; ffw_job q[FFW_Q]; int qh, qn, stop, busy[2]; atomic_int fail;
    void *evs[2], *evd[2]; uint8_t *h[2];
    z3w **zw; uint8_t **sl; const int *F; int FS, FZ, gpu;
    uint8_t *done;   /* the current slab's written field-shard rows [FZ / FS][ny] */
    ffring rg;       /* the slab's layout (a ring of rows with row flushes) */
    /* row flushes on their own thread (half the CPUs), so the scatter of window chunks never waits for compression: a queue of
       kind-2 jobs; the slab flush (kind 1) waits for it to drain */
    pthread_t rth; ffw_job rq[FFW_Q]; int rqh, rqn, rbusy, rstop, rnt;
} ffw_t;
static void *ffw_rows(void *arg) {
    ffw_t *W = arg;
    for (;;) {
        pthread_mutex_lock(&W->mu);
        while (!W->rqn && !W->rstop) pthread_cond_wait(&W->cv, &W->mu);
        if (!W->rqn) { pthread_mutex_unlock(&W->mu); break; }
        const ffw_job j = W->rq[W->rqh]; W->rqh = (W->rqh + 1) % FFW_Q; W->rqn--; W->rbusy = 1;
        pthread_mutex_unlock(&W->mu);
        if (!atomic_load(&W->fail) && ffz_flush(W->zw, W->sl, j.slab, W->F, W->FS, W->FZ, j.q0, j.q1, j.y0, j.y1, W->done, 0, W->rnt, &W->rg)) atomic_store(&W->fail, 1);
        if (W->rg.NR > 1 && !atomic_load(&W->fail)) ffring_release(&W->rg, W->sl, j.slab, (int64_t)j.y0 * W->FS / W->rg.YR, W->F[2]);   /* (a whole ring row) */
        pthread_mutex_lock(&W->mu); W->rbusy = 0; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
    return nullptr;
}
static void *ffw_main(void *arg) {
    ffw_t *W = arg;
    nn_init(W->gpu);
    for (;;) {
        pthread_mutex_lock(&W->mu);
        while (!W->qn && !W->stop) pthread_cond_wait(&W->cv, &W->mu);
        if (!W->qn) { pthread_mutex_unlock(&W->mu); break; }
        const ffw_job j = W->q[W->qh]; W->qh = (W->qh + 1) % FFW_Q; W->qn--;
        pthread_cond_broadcast(&W->cv);
        pthread_mutex_unlock(&W->mu);
        if (j.kind == 1) {   /* the slab is left: the row flusher drained, then the rows left (all CPUs) and the reset */
            pthread_mutex_lock(&W->mu); while (W->rqn || W->rbusy) pthread_cond_wait(&W->cv, &W->mu); pthread_mutex_unlock(&W->mu);
            if (!atomic_load(&W->fail) && ffz_flush(W->zw, W->sl, j.slab, W->F, W->FS, W->FZ, 0, W->FZ / W->FS, 0, 1 << 30, W->done, 1, 0, &W->rg)) atomic_store(&W->fail, 1);
            continue;
        }
        if (j.kind == 2) {   /* to the row flusher (its chunks are scattered: FIFO) */
            pthread_mutex_lock(&W->mu); while (W->rqn == FFW_Q) pthread_cond_wait(&W->cv, &W->mu);
            W->rq[(W->rqh + W->rqn) % FFW_Q] = j; W->rqn++; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
            continue;
        }
        if (W->rg.NR > 1) {   /* the chunk's ring slot: taken once its previous row is written (that row's flush is queued ahead: FIFO) */
            const int64_t rr = j.c1 / W->rg.YR, g = j.slab * W->rg.nRR + rr; const int s = (int)(rr % W->rg.NR);
            if ((j.c1 + j.ny - 1) / W->rg.YR != rr) { fprintf(stderr, "predict: --fit-zarr: a window's cells span two slab rows\n"); atomic_store(&W->fail, 1); }
            pthread_mutex_lock(&W->mu);
            while (W->rg.own[s] != g && W->rg.own[s] != -1 && !atomic_load(&W->fail)) pthread_cond_wait(&W->cv, &W->mu);
            if (W->rg.own[s] == -1) W->rg.own[s] = g;
            pthread_mutex_unlock(&W->mu);
        }
        nn_event_sync(W->evd[j.b]);
        const uint8_t *hb = W->h[j.b];
        if (!atomic_load(&W->fail)) for (int a = 0; a < j.nz; a++) for (int b = 0; b < j.ny; b++) for (int k = 0; k < ffz_np; k++)
            memcpy(ffz_row(W->sl[k], &W->rg, W->F[2], (int)(j.cz + a - j.slab * W->FZ), (int64_t)j.c1 + b) + j.c2, hb + k * j.nl + ((size_t)a * j.ny + b) * j.nx, (size_t)j.nx);
        pthread_mutex_lock(&W->mu); W->busy[j.b] = 0; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
    return nullptr;
}
static void ffw_push(ffw_t *W, ffw_job j) {
    pthread_mutex_lock(&W->mu);
    while (W->qn == FFW_Q) pthread_cond_wait(&W->cv, &W->mu);
    W->q[(W->qh + W->qn) % FFW_Q] = j; W->qn++;
    pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
}
static int ffw_acquire(ffw_t *W) {   /* a free staging buffer (its previous copy scattered) */
    pthread_mutex_lock(&W->mu);
    int b;
    for (;;) { b = !W->busy[0] ? 0 : !W->busy[1] ? 1 : -1; if (b >= 0) break; pthread_cond_wait(&W->cv, &W->mu); }
    W->busy[b] = 1;
    pthread_mutex_unlock(&W->mu);
    return b;
}
static int ffw_finish(ffw_t *W, pthread_t th) {
    pthread_mutex_lock(&W->mu); W->stop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    pthread_join(th, nullptr);
    pthread_mutex_lock(&W->mu); W->rstop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    pthread_join(W->rth, nullptr);
    return atomic_load(&W->fail);
}

static int writer_finish(writer_t *W, pthread_t th) {
    pthread_mutex_lock(&W->mu);
    for (int k = 0; k < W->nb; k++) while (W->pending[k]) pthread_cond_wait(&W->cv, &W->mu);   /* every buffer of the ring written */
    W->stop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    pthread_join(th, nullptr); return atomic_load(&W->failed);
}
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* --occupancy: any nonzero voxel of the coarse level (factor f, read over [o, o + n)) under the level region [g0, g1), one coarse voxel of margin */
static int occ_any(const uint8_t *ob, const int64_t o[3], const int64_t n[3], int f, const int64_t g0[3], const int64_t g1[3]) {
    int64_t a[3], b[3];
    for (int d = 0; d < 3; d++) {
        a[d] = g0[d] / f - 1 - o[d]; b[d] = (g1[d] + f - 1) / f + 1 - o[d];
        if (a[d] < 0) a[d] = 0;
        if (b[d] > n[d]) b[d] = n[d];
        if (b[d] <= a[d]) return 0;
    }
    for (int64_t z = a[0]; z < b[0]; z++) for (int64_t y = a[1]; y < b[1]; y++) {
        const uint8_t *r = ob + ((size_t)z * n[1] + y) * n[2];
        for (int64_t x = a[2]; x < b[2]; x++) if (r[x]) return 1;
    }
    return 0;
}
static int parse_grid_origin(const char *text, int64_t origin[3]) {
    for (int d = 0; d < 3; d++) {
        char *end; errno = 0;
        long long value = strtoll(text, &end, 10);
        if (errno || end == text || value < 0 || (d < 2 ? *end != ',' : *end != '\0')) return 1;
        origin[d] = value;
        text = end + (d < 2);
    }
    return 0;
}

static int sheet_write_shard(const char *out,const float *q,int shard,int64_t z,int64_t y,int64_t x) {
    char path[1600]; snprintf(path,sizeof path,"%s/winding/%lld.%lld.%lld.q.zst",out,(long long)z,(long long)y,(long long)x);
    size_t bytes=(size_t)shard*shard*shard*4,cap=ZSTD_compressBound(bytes); void *compressed=malloc(cap);
    if (!compressed) return -1;
    size_t n=ZSTD_compress(compressed,cap,q,bytes,3);
    if (ZSTD_isError(n)) { free(compressed); return -1; }
    FILE *f=fopen(path,"wb"); if (!f) { free(compressed); return -1; }
    int bad=fwrite(compressed,1,n,f)!=n; bad|=fclose(f)!=0; free(compressed); return bad?-1:0;
}

/* --split z: each window across two GPUs along z (src/split.h). GPU s holds the window planes [zoff, zoff + Dl) (its half plus
   2^(nlev-1) halo planes toward the other) and runs the CT-stem forward on them; the halves exchange halo planes and sum the
   GroupNorm statistics and the window's CT statistics, so the logits equal one GPU's. Each places the interior planes it owns into
   its device shard buffer and copies those planes to the host shard; side 0 hands the shard to the writer. The two job threads take
   turns on the host (split_run) and run the same sequence of collectives: the tile loop, the air decisions and the shard changes
   are identical on both. */
typedef struct { unet *u; uint8_t *ctd, *dsh, *own; uint32_t *ctm, *ctmi; void *stsc; float *dyd, *rows; double *red; int cuda_bad; } pside_t;
typedef struct {
    pside_t s[2];
    reader_t *rd; writer_t *WR; const tile_t *tiles; long ntiles;
    int W, halo, shard, Dl, H0, kss, h16, sign, sb, fail; int64_t ns[3], nss[3], bn[3], bo[3];
    const axis *ax; double scale, t0; long ndone, nskip;
    double pcut; int pcond;   /* task papyrus: the cutoff (raw CT units) of a conditioned net, else < 0; its conditioning (papyrus.h) */
} pjob_t;
static void psplit_flush(pjob_t *J, int side, int64_t qz, int64_t qy, int64_t qx) {   /* both sides: the owned planes to the host shard; side 0 submits */
    pside_t *S = &J->s[side];
    const size_t pl = (size_t)J->shard * J->shard;
    split_reduce(S->red, 1);   /* (side 0 has handed the previous shard to the writer: J->sb is this one's buffer; both have placed) */
    uint8_t *sbuf = J->WR->buf[J->sb];
    for (int z = 0; z < J->shard;) {
        if (!S->own[z]) { z++; continue; }
        int e = z; while (e < J->shard && S->own[e]) e++;
        nn_d2h(sbuf + (size_t)z * pl, S->dsh + (size_t)z * pl, (size_t)(e - z) * pl);
        z = e;
    }
    if (!side) for (int z = 0; z < J->shard; z++) if (!J->s[0].own[z] && !J->s[1].own[z]) memset(sbuf + (size_t)z * pl, 0, pl);   /* (before the barrier: side 1
                                                                                                          may reset its marks right after it) */
    split_reduce(S->red, 1);   /* both halves copied */
    if (side) return;
    if (writer_submit(J->WR, J->sb, qz, qy, qx)) J->fail = 1;
    J->sb = writer_next(J->WR, J->sb);
}
static void psplit_mark(uint8_t *own, int shard, int64_t oz, int64_t ez, int zlo, int zhi) {   /* shard planes oz + [zlo, zhi) within [0, ez) */
    for (int z = zlo; z < zhi; z++) { const int64_t g = oz + z; if (g >= 0 && g < ez && g < shard) own[g] = 1; }
}
static void psplit_job(int side, void *arg) {
    pjob_t *J = arg; pside_t *S = &J->s[side]; reader_t *rd = J->rd;
    const int W = J->W, Dl = J->Dl, H0 = J->H0, halo = J->halo, shard = J->shard;
    const int zoff = side ? W / 2 - H0 : 0, olo = side ? W / 2 : 0, ohi = side ? W : W / 2;   /* window planes held / owned */
    const int ilo = olo > halo ? olo : halo, ihi = ohi < W - halo ? ohi : W - halo;          /* owned interior planes */
    const size_t pl = (size_t)W * W, l3 = (size_t)Dl * pl, sh3 = (size_t)shard * shard * shard;
    const shape5 xs = {1, 4, Dl, W, W};
    int64_t cur = -1, so[3] = {0, 0, 0}, se[3] = {0, 0, 0}, sz = 0, sy = 0, sx = 0; int any = 0;
    for (long i = 0; i < J->ntiles; i++) {
        pthread_mutex_lock(&rd->mu);
        while (atomic_load(&rd->filled) <= i && !atomic_load(&rd->failed)) pthread_cond_wait(&rd->cv, &rd->mu);
        pthread_mutex_unlock(&rd->mu);
        const tile_t *t = &J->tiles[i];
        if (J->kss == 1 && t->shard != cur) {   /* new shard: the previous one out */
            if (cur >= 0 && any) psplit_flush(J, side, sz, sy, sx);
            cur = t->shard; sz = cur / (J->ns[1] * J->ns[2]); sy = (cur / J->ns[2]) % J->ns[1]; sx = cur % J->ns[2];
            so[0] = sz * shard; so[1] = sy * shard; so[2] = sx * shard;
            for (int d = 0; d < 3; d++) se[d] = so[d] + shard < J->bn[d] ? so[d] + shard : J->bn[d];
            nn_zero(S->dsh, sh3); memset(S->own, 0, (size_t)shard); any = 0;
        }
        const int bad = atomic_load(&rd->failed) && atomic_load(&rd->filled) <= i;
        double h[4] = {0, 0, 0, (double)(bad || J->fail || S->cuda_bad)};
        if (!bad) {   /* this half's planes up; the CT statistics over the planes it owns, summed over both */
            nn_h2d(S->ctd, rd->buf[i % NSLOT] + (size_t)zoff * pl, l3);
            size_t nz; double sum, sq;
            nn_pred_stats(S->ctd + (size_t)(olo - zoff) * pl, (size_t)(W / 2) * pl, S->stsc, &nz, &sum, &sq);
            h[0] = (double)nz; h[1] = sum; h[2] = sq;
        }
        nn_h2d(S->red, h, sizeof h); split_reduce(S->red, 4); nn_d2h(h, S->red, sizeof h);
        pthread_mutex_lock(&rd->mu); if (atomic_load(&rd->consumed) < i + 1) atomic_store(&rd->consumed, i + 1); pthread_cond_broadcast(&rd->cv); pthread_mutex_unlock(&rd->mu);
        if (h[3] > 0) { J->fail = J->fail ? J->fail : 2; return; }   /* (both halves see it at the same collective) */
        if (h[0] == 0) { if (!side) J->nskip++; continue; }
        const double w3 = (double)W * W * W, wmean = h[1] / w3, var = h[2] / w3 - wmean * wmean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
        if (J->pcut >= 0 && J->pcond == 1) unet_set_cond(S->u, (float)((J->pcut - wmean) / sd));   /* papyrus: the cutoff in this window's input units */
        const double mean = J->pcut >= 0 && J->pcond == 2 ? J->pcut : wmean;                        /* (or the input's zero) */
        const int64_t o[3] = {J->bo[0] + t->tz, J->bo[1] + t->ty, J->bo[2] + t->tx};
        for (int z = 0; z < Dl; z++) {   /* window origin relative to the axis at this half's slices */
            double cy, cx;
            axis_at(J->ax, (double)(o[0] + zoff + z) * J->scale, &cy, &cx);
            cy /= J->scale; cx /= J->scale;
            S->rows[z] = (float)((double)o[1] - cy); S->rows[Dl + z] = (float)((double)o[2] - cx);
        }
        nn_h2d(S->dyd, S->rows, 2 * (size_t)Dl * sizeof(float));
        nn_pred_ctmask(S->ctd, l3, S->ctm);   /* the forward overwrites ctd */
        if (S->ctmi != S->ctm) nn_pred_ctmask_t(S->ctd, l3, S->ctmi, nn_get_pred_iso());   /* --iso: the outputs' mask */
        pred_axis(S->u, J->ax->n > 0, Dl, S->rows, S->rows + Dl);
        const float *lg = unet_forward_ct(S->u, S->ctd, (float)mean, (float)(1.0 / sd), S->dyd, S->dyd + Dl, J->ax->n > 0 ? J->sign : 0, xs, J->h16);
        nn_set_logits_h16(unet_logits_h16(S->u));   /* (both halves build alike: the process-wide flag holds the same value) */
        if (J->kss > 1) {   /* every shard of this super-shard from the one window */
            const int64_t SZ = t->shard / (J->nss[1] * J->nss[2]), SY = (t->shard / J->nss[2]) % J->nss[1], SX = t->shard % J->nss[2], k = J->kss;
            for (int64_t qz = SZ * k; qz < SZ * k + k && qz < J->ns[0]; qz++) for (int64_t qy = SY * k; qy < SY * k + k && qy < J->ns[1]; qy++)
            for (int64_t qx = SX * k; qx < SX * k + k && qx < J->ns[2]; qx++) {
                const int64_t qo[3] = {qz * shard, qy * shard, qx * shard}, qe[3] = {qo[0] + shard < J->bn[0] ? qo[0] + shard : J->bn[0], qo[1] + shard < J->bn[1] ? qo[1] + shard : J->bn[1], qo[2] + shard < J->bn[2] ? qo[2] + shard : J->bn[2]};
                nn_zero(S->dsh, sh3); memset(S->own, 0, (size_t)shard);
                nn_pred_place_mz(lg, S->ctmi, W, halo, olo, ohi, zoff, (int)(t->tz - qo[0]), (int)(t->ty - qo[1]), (int)(t->tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, S->dsh);
                psplit_mark(S->own, shard, t->tz - qo[0], qe[0] - qo[0], ilo, ihi);
                psplit_flush(J, side, qz, qy, qx);
            }
        } else {
            nn_pred_place_mz(lg, S->ctmi, W, halo, olo, ohi, zoff, (int)(t->tz - so[0]), (int)(t->ty - so[1]), (int)(t->tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]), shard, S->dsh);
            psplit_mark(S->own, shard, t->tz - so[0], se[0] - so[0], ilo, ihi);
            any = 1;
        }
        if (nn_check()) S->cuda_bad = 1;
        if (!side) { J->ndone++; fprintf(stderr, "\rwindow %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, J->ntiles, J->ndone, J->nskip, now() - J->t0); }
    }
    if (J->kss == 1 && cur >= 0 && any) psplit_flush(J, side, sz, sy, sx);
}

/* the attention stage's scroll axis (unet_cfg.attn_axis radial / attn_pe scroll): the window's axis rows (dy[z] = window origin y - axis
   centre y at plane z, likewise dx) as the record of unet_set_attn_axis (the axis along window z); g_pax_div: the window voxels per network
   voxel, 0 when the net takes no axis */
static int g_pax_div;
static void pred_axis(unet *u, int valid, int D, const float *dy, const float *dx) {
    if (g_pax_div <= 0) return;
    float *r = malloc(UNET_AXR(D) * sizeof *r);
    if (!r) abort();
    r[0] = 0.f; r[1] = valid ? 1.f : 0.f; r[2] = (float)D; r[3] = (float)g_pax_div;
    for (int z = 0; z < D; z++) { r[4 + 2 * z] = -dy[z]; r[5 + 2 * z] = -dx[z]; }
    unet_set_attn_axis(u, r, 1, UNET_AXR(D));
    free(r);
}
/* window memory (unet_cfg.attn_mem): the previous predicted window's memory rows (whole-window passes only: TTA and split passes run
   without, as chain starts do in training) when it lies within a window size of this one */
static float *g_pmem; static int64_t g_pmo[3]; static int g_pmv;
static void pred_mem_in(unet *u, const int64_t o[3], int W) {
    if (!unet_cfg_of(u)->attn_mem) return;
    int near = g_pmv;
    for (int d = 0; d < 3 && near; d++) near = llabs(o[d] - g_pmo[d]) <= W;
    if (!near) { unet_set_attn_mem(u, nullptr, nullptr, 0); return; }
    const float feat[4] = {(float)(o[0] - g_pmo[0]) / W, (float)(o[1] - g_pmo[1]) / W, (float)(o[2] - g_pmo[2]) / W, 1.f};
    unet_set_attn_mem(u, g_pmem, feat, 1);
}
static void pred_mem_out(unet *u, const int64_t o[3]) {
    int M, C; const float *out = unet_cfg_of(u)->attn_mem ? unet_attn_mem_out(u, &M, &C) : nullptr;
    if (!out) return;
    if (!g_pmem) g_pmem = nn_malloc((size_t)M * C * 4);
    nn_d2d(g_pmem, out, (size_t)M * C * 4); memcpy(g_pmo, o, sizeof g_pmo); g_pmv = 1;
    unet_set_attn_mem(u, nullptr, nullptr, 0);   /* (the next passes: TTA, split, until the next window sets its own) */
}

/* scan fusion (unet_cfg.view_P): per window the view crop centred on it (src/view.h; --view-level L on this scan's grid, --view-map
   REG.json for a registered partner scan, else this scan itself at level L), as in training. Use the training window size: the crop
   covers the window's centre whatever the window */
static sview *g_pv; static z3 *g_pvself; static uint8_t *g_pvh, *g_pvd, *g_pvsave; static int g_pvP, g_pvlev;
static int pred_view_open(const unet_cfg *cfg, const char *root, const char *key, int level, double um, const char *cache, const char *lvs, const char *map, int split) {
    const int lv = atoi(lvs);
    if (split) { fprintf(stderr, "predict: a scan-fusion checkpoint (view_P %d) does not run --split\n", cfg->view_P); return -1; }
    if (lv < 0 || lv > level) { fprintf(stderr, "predict: --view-level %d must lie in 0..--level %d\n", lv, level); return -1; }
    if (!(g_pv = sview_open(map, lv, um, cache))) return -1;
    if (!map) { store *s = store_open(root); g_pvself = s ? pyramid_open_level(s, key, lv, um, cache) : nullptr; if (!g_pvself) { fprintf(stderr, "predict: cannot open level %d for the view\n", lv); return -1; } }
    g_pvP = cfg->view_P; g_pvlev = level;
    const size_t n = (size_t)g_pvP * g_pvP * g_pvP;
    g_pvh = malloc(n); g_pvd = nn_malloc(n); g_pvsave = nn_malloc(n);
    if (!g_pvh || !g_pvd || !g_pvsave) { fprintf(stderr, "predict: no memory for the view crops\n"); return -1; }
    fprintf(stderr, "predict: scan fusion: %d^3 view crops at level %d (%s)\n", g_pvP, lv, map ? map : "this scan");
    return 0;
}
static int pred_view_set(unet *u, const int64_t o[3], int Wz, int W) {   /* the window's crop (level-`level` corner o) */
    if (!g_pv) return 0;
    int64_t vo[3]; const int64_t n[3] = {Wz, W, W};
    sview_origin(g_pvlev, g_pv->lv, o, n, g_pvP, vo);
    if (sview_read(g_pv, g_pvself, vo, g_pvP, g_pvh)) { fprintf(stderr, "predict: view read failed\n"); return -1; }
    nn_h2d(g_pvd, g_pvh, (size_t)g_pvP * g_pvP * g_pvP);
    unet_set_attn_view(u, g_pvd, 1);
    return 0;
}

/* predict --tta: one augmented pass of a window (src/tta.h): the saved CT window through T into the network's input buffer, the
   mirrored axis rows, the forward, the logits back through T (in place); returns them */
typedef struct { unet *u; uint8_t *ctd; const uint8_t *ctsave; float mean, isd; const float *dyo, *dxo; float *dyt, *dyd; int axis; shape5 xs; int h16, D, W, nch, ch_normal; } tta_ctx;
static const float *tta_pass(const tta_ctx *T, int code) {
    nn_tta_ct(T->ctsave, T->ctd, T->D, T->W, code);
    if (g_pv) { nn_tta_ct(g_pvsave, g_pvd, g_pvP, g_pvP, code); unet_set_attn_view(T->u, g_pvd, 1); }   /* (the view crop through the same T) */
    tta_axis(code, T->D, T->W, T->dyo, T->dxo, T->dyt, T->dyt + T->W);
    nn_h2d(T->dyd, T->dyt, 2 * (size_t)T->W * sizeof(float));
    pred_axis(T->u, T->axis != 0, T->D, T->dyt, T->dyt + T->W);
    const float *lg = unet_forward_ct(T->u, T->ctd, T->mean, T->isd, T->dyd, T->dyd + T->W, T->axis, T->xs, T->h16);
    const int lh = unet_logits_h16(T->u); nn_set_logits_h16(lh);
    int src[32], sgn[32]; tta_channels(code, T->ch_normal, T->nch, src, sgn);
    nn_tta_logits((void *)lg, lh, T->D, T->W, code, T->nch, src, sgn);
    return lg;
}

int cmd_predict(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm predict <ckpt> <root> <ct-group-key> <out-dir> --um U [--level 0] [--box z,y,x,nz,ny,nx] [--window W] [--act-mx8 1]\n"
                        "       [--halo H] [--shard 512] [--gpu 0] [--cache DIR] [--axis umbilicus.json] [--levels 4] [--q 8] [--threads 16]\n"
                        "       [--prec 0|1|2|3|4] [--policy enc0=1,...] [--f16 0|1] [--fp4 0|1] [--input-mx 0|1] [--input-prec 0|4|8] [--ema 1]   EMA or current weights (0)\n"
                        "       [--iso T]   probability outputs 0 where CT < T (the scan's void cutoff; the fit's material mask stays CT > 0)\n"
                        "       [--ema-blend A]   a x final weights + (1 - a) x EMA (tail-average blend, e.g. 0.35)\n"
                        "       [--gn-stats stored|legacy]   defaults to the checkpoint's normalization contract\n"
                        "       [--reference reference.json]   required for a surface_winding checkpoint, native level 0\n"
                        "       [--grid-origin z,y,x]   anchor tile interiors in selected-level CT voxels, independent of the output box\n"
                        "       [--input-gain G] [--input-bias B]   CT = G * zscore(CT) + B; radial inputs and material mask unchanged\n"
                        "       [--gpus 0,1]   one worker per GPU over the shards of the same output (needs --box)\n"
                        "       [--split z --gpus 0,1]   both GPUs on every window, one z half each (window a multiple of 2^levels; default window shard + 128, halo 64)\n"
                        "       [--feather B]   seam-free single pass: each window border blended over B voxels (sin^2 ramp, B <= halo) from the earlier\n"
                        "                       windows' halo predictions; window - 2 halo = k * shard on the shard grid; --gpus deal contiguous z layers\n"
                        "       [--channel 0]   output channel written (band_affinity checkpoints: 1..6 = affinities z1 y1 x1 z8 y8 x8)\n"
                        "       [--fit-fields DIR] [--fit-h 8]   task unwrap checkpoints: also the winding fitter's input fields of the box (ufsm fit DIR)\n"
                        "       [--fit-zarr DIR]                 the same fields as zarr arrays, flushed slab by slab (whole volumes)\n"
                        "       [--tta K]   the fit fields averaged over K window symmetries (z / y / x flips, y <-> x swap; 1..12, src/tta.h)\n"
                        "       [--view-level L] [--view-map REG.json]   scan-fusion checkpoints (view_P): the view crop per window from level L of this scan\n"
                        "                   or, with a map, the registered partner scan on this scan's level-L grid (use the training window size)\n"
                        "       [--fit-w4-d 32]   task unwrap2 checkpoints: the winding mod 4 relative planes dz dy dx dq over this many voxels (<= the halos)\n"
                        "       [--fit-ct 1]                     with them the CT planes ct / cs (fit --w-ct / --w-cspacing; ufsm ct-fields adds them later)\n"
                        "       [--no-out 1]                     skip writing the prediction volume's shards (e.g. only the --fit-zarr fields)\n"
                        "       [--fit-q 8] [--fit-q-phase 4]   volcomp quantiser of the --fit-zarr arrays (0 lossless; the phase planes pc / ps: --fit-q-phase)\n"
                        "       [--radial-sign 1|-1]   -1 reverses the direction input (the CT is unchanged): a recto model then marks the verso face\n"
                        "       [--gn-save FILE]   pool the GroupNorm statistics of every window processed into FILE (calibration)\n"
                        "       [--gn-frozen FILE]   use FILE's pooled GroupNorm statistics instead of each window's (no dependence on the window)\n"
                        "       [--papyrus-cut T]   task papyrus nets conditioned on the cutoff: papyrus where the smoothed CT exceeds T (raw CT units)\n"
                        "       [--full-grid 1]   the output spans the CT level's whole grid; --box (rounded out to shards) selects the shards computed\n"
                        "       [--resume 1]   skip the shards listed in <out>/predict-done-<level>.txt (appended as shards are written)\n"
                        "       [--occupancy L]   skip shards that are all zero (air) on the coarser CT level L, e.g. 5\n");
        fprintf(stderr, "  Default window: k shards + 2 halo (one window per k^3 shards) for the largest k <= 4 the box and the GPU hold, else one per shard,\n"
                        "  else 288 with halo 8; the halo is 16, or 32 when the window must be a multiple of 64 (7 levels). For exact shard tiling, window - 2*halo should\n"
                        "  divide the shard size: 544 with halo 16 gives 512; 288 with halo 16 gives 256. Validate window and halo with the checkpoint.\n"
                        "  Window - 2*halo = k * shard (k >= 2) with the box on that grid computes one window per k^3 shards (big GPUs: fewer, larger windows).\n"
                        "  Inference memory: UFSM_INFER_SLAB=N (default 64) runs the level-0/1 convs in place in N-plane slabs; UFSM_INFER_OFFLOAD=1 (default) keeps\n"
                        "  the coarse levels in the level-0 buffer's tail while it waits on the host: ~21.3 B per window voxel with a 32-wide level 0 (0: ~27;\n"
                        "  slabs 0: ~43.5), ~10.1 with the 16-wide production net (1088^3 on a 16 GB GPU).\n");
        return 2;
    }
    const char *ckpt = argv[2], *root = argv[3], *key = argv[4], *out = argv[5];
    { int restore_checkpoint(const char *path, int *split, double *fine_um); int restore_predict(int argc, char **argv);   /* src/restore_cli.c */
      if (restore_checkpoint(ckpt, nullptr, nullptr)) return restore_predict(argc, argv); }   /* task restore: the SR path */
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), W = atoi(opt(argc, argv, "--window", "288")), halo = atoi(opt(argc, argv, "--halo", "8"));
    int shard = atoi(opt(argc, argv, "--shard", "512")), gpu = atoi(opt(argc, argv, "--gpu", "0")), nlev = atoi(opt(argc, argv, "--levels", "4"));
    const double input_gain = atof(opt(argc, argv, "--input-gain", "1")), input_bias = atof(opt(argc, argv, "--input-bias", "0"));
    if (!isfinite(input_gain) || input_gain <= 0 || !isfinite(input_bias)) { fprintf(stderr, "predict: --input-gain must be finite and positive; --input-bias finite\n"); return 2; }
    const int input_affine = input_gain != 1 || input_bias != 0;
    char input_attrs[128] = "";
    if (input_affine) snprintf(input_attrs, sizeof input_attrs, ",\"input_gain\":%.9g,\"input_bias\":%.9g", input_gain, input_bias);
    const int win_given = *opt(argc, argv, "--window", "") || *opt(argc, argv, "--halo", "");
    /* --window-z WZ [--halo-z HZ]: slab windows WZ x W x W (train --Pz nets; HZ default WZ / 4), the plain GPU path only */
    int Wz = atoi(opt(argc, argv, "--window-z", "0")), hz = atoi(opt(argc, argv, "--halo-z", "-1"));
    const int slab = Wz > 0 && Wz != W;
    if (Wz <= 0) Wz = W;
    if (hz < 0) hz = slab ? Wz / 4 : halo;
    /* --aug-OP-p P [--aug-p P] [--aug-strength S] [--aug-seed N]: degrade every CT window with train's scanner augmentation
       (src/scan_augment.h; seam off) before the network, seeded by the window's origin: robustness of a checkpoint to
       scanner-domain shifts (blur, noise, rings, quantisation, ...) on held-out boxes */
    scan_aug_cfg dg = {.strength = (float)atof(opt(argc, argv, "--aug-strength", "1")), .drop_ignore = 1};
    for (int i = 0; i < SA_N; i++) { char f[48]; snprintf(f, sizeof f, "--aug-%s-p", scan_aug_names[i]); dg.p[i] = (float)atof(opt(argc, argv, f, opt(argc, argv, "--aug-p", "0"))); }
    for (int i = 0; i < SA_N; i++) { char f[48]; snprintf(f, sizeof f, "--aug-%s-s", scan_aug_names[i]); dg.s[i] = (float)atof(opt(argc, argv, f, "0")); }
    if (*opt(argc, argv, "--aug-hist", "") && scan_aug_load_hist(opt(argc, argv, "--aug-hist", ""), &dg)) { fprintf(stderr, "predict: --aug-hist: cannot read the file\n"); return 2; }
    const int degrade = scan_aug_enabled(&dg); const uint64_t dg_seed = strtoull(opt(argc, argv, "--aug-seed", "1"), nullptr, 10);
    int nthreads = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    const int use_ema = atoi(opt(argc, argv, "--ema", "1")) != 0;
    const char *blend_s = opt(argc, argv, "--ema-blend", nullptr);   /* tail-average blend: a x final weights + (1 - a) x EMA (modded-nanogpt 0.35) */
    const float ema_blend = blend_s ? (float)atof(blend_s) : -1.f;
    if (blend_s && !(ema_blend >= 0.f && ema_blend <= 1.f)) { fprintf(stderr, "--ema-blend: 0..1\n"); return 2; }
    const double radial_sign = atof(opt(argc, argv, "--radial-sign", "1"));   /* -1: reversed direction input (a recto model then marks the verso face) */
    if (radial_sign != 1 && radial_sign != -1) { fprintf(stderr, "predict: --radial-sign must be 1 or -1\n"); return 2; }
    const int out_ch = atoi(opt(argc, argv, "--channel", "0"));   /* output channel to write (band_affinity: 1..6 = affinities) */
    /* --fit-fields DIR (task unwrap checkpoints: recto 0, ink 1, phase 2-3, normal 4-9, density 10): every head computed, each
       window's interior pooled on the device into the winding fitter's fields (h native voxels per field voxel, --fit-h 8) of the
       box, written to DIR at the end (src/fit.h fit_fields_save). One GPU, the CT window in its own buffer (no aliasing). */
    const char *ffd = opt(argc, argv, "--fit-fields", nullptr); const int ffh = atoi(opt(argc, argv, "--fit-h", "8"));
    const char *ffz = opt(argc, argv, "--fit-zarr", nullptr);   /* the same fields of a whole volume: 10-16 uint8 zarr arrays (ffz_names), flushed slab by slab */
    /* --fit-q Q: volcomp quantiser of the --fit-zarr arrays (0 lossless, 1..255 lossy DCT; default 8); --fit-q-phase: the phase planes
       pc / ps (default 4; was 0, lossless: ~90% of a 21-30 GB whole-scroll store; q 4 makes the phase 28-31x smaller, phase error
       median 0.007 turn; slab fits + fit-join + wrap-check unchanged on PHerc0490B / PHerc1299 / Paris 4 crops, Paris 4 GT ids .1666 vs
       .1668, pairs -0.1..-0.2 pt; ufsm-fieldq-2026-10-30). Old stores load unchanged. */
    const float ffq = (float)atof(opt(argc, argv, "--fit-q", "8")), ffqp = (float)atof(opt(argc, argv, "--fit-q-phase", "4"));
    if (!(ffq >= 0 && ffq <= 255 && ffqp >= 0 && ffqp <= 255)) { fprintf(stderr, "predict: --fit-q / --fit-q-phase 0..255\n"); return 2; }
    if (ffd && ffz) { fprintf(stderr, "predict: --fit-fields or --fit-zarr\n"); return 2; }
    if ((ffd || ffz) && (ffh < 1 || ffh > 32)) { fprintf(stderr, "--fit-h 1..32\n"); return 2; }   /* (inference aliasing stays on: the fit pooling reads
                                                                                                       the material mask from the CT bit plane; it was off: ~80 instead of ~45 B per window voxel) */
    if (ffz) ffd = ffz;   /* the shared checks and settings below */
    /* --fit-ct 1: also the CT planes ct (mean CT per cell) and cs (CT sheet spacing: papyrus peaks along the network normal, src/fit_terms.h
       fit_ct_spacing) for fit --w-ct / --w-cspacing; the CT window is copied before the forward when the network aliases it */
    const int ffct = atoi(opt(argc, argv, "--fit-ct", "0")) != 0;
    if (ffct && !ffd) { fprintf(stderr, "predict: --fit-ct needs --fit-fields or --fit-zarr\n"); return 2; }
    const int ffna = ffct ? 14 : 12;   /* accumulator planes */
    const int ffzm = ffz != nullptr;
    /* --tta K: the fit fields averaged over K window symmetries (src/tta.h; identity first); the probability output is the last pass */
    const int tta = atoi(opt(argc, argv, "--tta", "1"));
    const int fea = atoi(opt(argc, argv, "--feather", "0"));   /* --feather B (above) */
    if (tta < 1 || tta > TTA_MAX || (tta > 1 && !ffd)) { fprintf(stderr, "predict: --tta 1..%d, with --fit-fields / --fit-zarr\n", TTA_MAX); return 2; }
    if (tta > 1 && ffct) { fprintf(stderr, "predict: --tta with --fit-ct is not supported (the CT planes pool one pass); add them later with ufsm ct-fields\n"); return 2; }
    float q = (float)atof(opt(argc, argv, "--q", "8"));
    const int iso = atoi(opt(argc, argv, "--iso", "0"));   /* the void cutoff: probability outputs 0 where CT < iso (the fit keeps CT > 0) */
    if (iso < 0 || iso > 255) { fprintf(stderr, "--iso: 0..255\n"); return 2; }
    nn_set_pred_iso(iso);
    const char *cache = opt(argc, argv, "--cache", nullptr), *axisf = opt(argc, argv, "--axis", nullptr);
    const char *grid = opt(argc, argv, "--grid-origin", nullptr);
    int64_t grid_origin[3] = {0, 0, 0}; char grid_attrs[128] = "";
    if (grid && parse_grid_origin(grid, grid_origin)) { fprintf(stderr, "--grid-origin needs three nonnegative integer coordinates\n"); return 2; }
    if (grid) snprintf(grid_attrs, sizeof grid_attrs, ",\"grid_origin_zyx\":[%lld,%lld,%lld]", (long long)grid_origin[0], (long long)grid_origin[1], (long long)grid_origin[2]);
    long long boxv[6]; const char *box = opt(argc, argv, "--box", nullptr); char tail;
    if (!isfinite(um) || um <= 0 || W <= 0 || halo < 0 || halo >= W / 2 || shard < 128 || shard % 128 ||
        level < 0 || level > 20 || nlev < 1 || nlev > 20 || nthreads < 1 || !isfinite(q) || q < 0 ||
        (box && (sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld%c", &boxv[0], &boxv[1], &boxv[2], &boxv[3], &boxv[4], &boxv[5], &tail) != 6 ||
                 boxv[0] < 0 || boxv[1] < 0 || boxv[2] < 0 || boxv[3] <= 0 || boxv[4] <= 0 || boxv[5] <= 0))) {
        fprintf(stderr, "invalid prediction geometry: require positive --um/window/box, 0 <= halo < window/2, shard a multiple of 128, and valid levels/threads/q\n"); return 2;
    }
    if (box) for (int d = 0; d < 3; d++) {
        int margin = W > shard ? W : shard;
        if (boxv[d + 3] > INT64_MAX - margin || boxv[d] > INT64_MAX - margin - boxv[d + 3]) {
            fprintf(stderr, "prediction box exceeds the coordinate range\n"); return 2;
        }
    }
    unet_cfg cfg; int step = 0; checkpoint_runtime runtime;
    int embedded = checkpoint_runtime_read(ckpt, &runtime);
    char geometry_sha[65],reference_sha[65]; int sheet_task=sheet_checkpoint(ckpt,geometry_sha,reference_sha);
    sheet_dataset *sheet=nullptr; const char *geometry=opt(argc,argv,"--geometry",nullptr);
    const char *reference=opt(argc,argv,"--reference",nullptr);
    if (sheet_task<0 || (sheet_task && !geometry && !reference) || (!sheet_task && (geometry || reference)) || (geometry && reference)) { fprintf(stderr,"winding prediction requires its --reference or --geometry contract; legacy models do not accept it\n"); return 2; }
    if (sheet_task) {
        sheet=reference?sheet_load_reference(reference):sheet_load(geometry);
        if (reference && sheet) strcpy(sheet->manifest_sha,geometry_sha);
        if (!sheet || strcmp(geometry_sha,sheet->manifest_sha) || strcmp(reference_sha,sheet->reference_sha) || level!=0) { fprintf(stderr,"winding reference mismatch or non-native level\n"); return 2; }
    }
    if (embedded < 0 || unet_peek(ckpt, &cfg, &step) || cfg.nlev < 1 || cfg.nlev > UNET_MAXLEV) { fprintf(stderr, "cannot read checkpoint settings: %s\n", ckpt); return 1; }
    nn_set_nlev(cfg.nlev);   /* layer names of the embedded / requested policy */
    const int half = cfg.half_res, hdiv = unet_hr_div(half), hmode = unet_hr_mode(half), nout = hmode == 2 ? cfg.cout / 8 : cfg.cout;   /* low-resolution
                                                                     network (nn_lowres_*): hdiv window voxels per network voxel; pack has 8 logits per output */
    g_pax_div = unet_cfg_axis(&cfg) ? hdiv : 0;
    if (out_ch < 0 || out_ch >= nout) { fprintf(stderr, "predict: --channel %d outside the checkpoint's %d outputs\n", out_ch, nout); return 2; }
    const char *sweep_out = opt(argc, argv, "--cond-sweep", nullptr);   /* train --cond checkpoints: conditioned inference (src/sweep.h) */
    if (sweep_out && (cfg.cin != 7 || cfg.cout < 15 || half || sheet_task || ffd)) { fprintf(stderr, "predict: --cond-sweep needs a train --cond checkpoint (7 inputs, 15 outputs), no --fit-fields\n"); return 2; }
    { const char *gl = opt(argc, argv, "--gpus", nullptr);   /* --gpus a,b (workers, not --split): each would write (and fill) the whole field arrays */
      if (ffd && gl && strchr(gl, ',') && !opt(argc, argv, "--split", nullptr)) {
          fprintf(stderr, "predict: --gpus with --fit-zarr / --fit-fields is not supported (each worker writes the whole field set): run one predict per GPU on z-halves "
                          "of the box (--grid-origin anchored) and join them with scripts/unroll/fields_cat.sh\n");
          return 2;
      } }
    if (ffd && (cfg.cout < 11 || half || sheet_task)) { fprintf(stderr, "predict: --fit-fields needs a task unwrap checkpoint (11+ outputs) at full resolution\n"); return 2; }
    /* the head layout: task unwrap (recto 0, ink 1, phase 2-3, normal 4-9, density 10) or unwrap2 (phase 0-1, normal 2-7, density 8,
       winding mod 4 9-10: no recto; the winding mod 4 pooled into the relative planes dz dy dx dq over --fit-w4-d voxels) */
    const int ffu2 = ffd && checkpoint_unwrap_version(ckpt) == 2;
    const int ff_rec = ffu2 ? -1 : 0, ff_ph = ffu2 ? 0 : 2, ff_nrm = ffu2 ? 2 : 4, ff_den = ffu2 ? 8 : 10, ff_w4 = ffu2 ? 9 : -1;
    int ffw4d = ffu2 ? atoi(opt(argc, argv, "--fit-w4-d", "32")) : 0;
    float ffds = 32.f; int ffdc = 0;   /* the checkpoint's density code: min(1, density_scale rho), or x / (1 + x) (density_code 1) */
    if (ffd) { FILE *cf = fopen(ckpt, "rb"); char *hb = malloc(1 << 16); size_t nr = cf ? fread(hb, 1, (1 << 16) - 1, cf) : 0; if (cf) fclose(cf); hb[nr] = 0;
               char *e = memchr(hb, '\n', nr); if (e) *e = 0; const char *q = strstr(hb, "\"density_scale\":"); if (q) ffds = (float)atof(q + 16);
               q = strstr(hb, "\"density_code\":"); if (q) ffdc = atoi(q + 15); free(hb); }
    int pap_cond = 0; const int pap = papyrus_checkpoint(ckpt, &pap_cond) == 1;   /* task papyrus: the conditioned nets take the cutoff per window */
    const double pcut = pap && pap_cond ? atof(opt(argc, argv, "--papyrus-cut", "-1")) : -1;
    if (pap && pap_cond && !(pcut >= 0 && pcut <= 255)) { fprintf(stderr, "predict: %s is conditioned on the papyrus cutoff: --papyrus-cut T (raw CT units 0..255, e.g. 80)\n", ckpt); return 2; }
    if (*opt(argc, argv, "--papyrus-cut", "") && !(pap && pap_cond)) { fprintf(stderr, "predict: --papyrus-cut needs a cutoff-conditioned papyrus checkpoint\n"); return 2; }
    char content[96]; if (pap && pcut >= 0) snprintf(content, sizeof content, "papyrus probability at cutoff %g", pcut); else snprintf(content, sizeof content, "%s probability", pap ? "papyrus" : "recto");
    const char *gn_stats = opt(argc, argv, "--gn-stats", embedded && runtime.gn_stored ? "stored" : "legacy");
    if (strcmp(gn_stats, "stored") && strcmp(gn_stats, "legacy")) { fprintf(stderr, "--gn-stats must be stored or legacy\n"); return 2; }
    nn_set_gn_stored(!strcmp(gn_stats, "stored"));
    if (win_given && W % (hdiv << (cfg.nlev - 1))) { fprintf(stderr, "window must be divisible by %d\n", hdiv << (cfg.nlev - 1)); return 2; }   /* (the default is chosen below) */
    const int override = *opt(argc, argv, "--prec", "") || *opt(argc, argv, "--fp4", "") || *opt(argc, argv, "--policy", "");
    int precision = *opt(argc, argv, "--prec", "") ? atoi(opt(argc, argv, "--prec", "1")) : embedded ? runtime.prec : 1;
    if (precision < 0 || precision > 4) { fprintf(stderr, "--prec must be 0..4\n"); return 2; }
    nn_set_prec(precision);
    if (nn_set_prec_policy(opt(argc, argv, "--policy", embedded && !override ? runtime.policy : ""))) return 2;
    if (embedded) nn_set_f16(runtime.f16);
    if (*opt(argc, argv, "--f16", "")) nn_set_f16(atoi(opt(argc, argv, "--f16", "0")) != 0);
    /* default --fp4 follows how the checkpoint was trained: models trained with fp4 storage (precision.txt next to the
       checkpoint says act_mx4 1) are calibrated for it; 16-bit-trained models keep MX-fp8 storage (fp4 storage shifts
       their best cutoff) */
    int ckfp4 = embedded ? runtime.act_mx4 : 0;
    if (!embedded) { char mp[1400]; snprintf(mp, sizeof mp, "%s", ckpt); char *sl = strrchr(mp, '/'); if (sl) sl[1] = 0; else mp[0] = 0;
      strncat(mp, "precision.txt", sizeof mp - strlen(mp) - 1); FILE *mf = fopen(mp, "r");
      if (mf) { char ln[4096]; while (fgets(ln, sizeof ln, mf)) if (strstr(ln, "act_mx4 1")) ckfp4 = 1; fclose(mf); }
      if (ckfp4 && !*opt(argc, argv, "--fp4", "")) fprintf(stderr, "checkpoint trained with fp4 storage (%s): --fp4 1\n", mp); }
    int mx4 = atoi(opt(argc, argv, "--fp4", ckfp4 ? "1" : "0")) != 0;
    int mx8 = embedded ? runtime.act_mx8 : 1;
    if (*opt(argc, argv, "--fp4", "")) { mx8 = !mx4; if (mx4 && !*opt(argc, argv, "--prec", "")) nn_set_prec(3); }
    else if (!embedded && mx4 && !*opt(argc, argv, "--prec", "")) nn_set_prec(3);
    if (*opt(argc, argv, "--act-mx4", "")) mx4 = atoi(opt(argc, argv, "--act-mx4", "0")) != 0;
    if (*opt(argc, argv, "--act-mx8", "")) { mx8 = atoi(opt(argc, argv, "--act-mx8", "0")) != 0; if (mx8 && !*opt(argc, argv, "--act-mx4", "")) mx4 = 0; }
    unet_set_act_mx4(mx4); unet_set_act_mx8(mx8);
    unet_set_grad_mx8(0);   /* inference has no gradient buffers; restore input precision independently */
    int input = *opt(argc, argv, "--input-mx", "") ? atoi(opt(argc, argv, "--input-mx", "0")) : embedded ? runtime.input_mx : ufsm_env_on("UFSM_XIN_MX");
    if (input < 0 || input > 1) { fprintf(stderr, "--input-mx must be 0 or 1\n"); return 2; }
    int input_prec = *opt(argc, argv, "--input-prec", "") ? atoi(opt(argc, argv, "--input-prec", "0")) : embedded ? runtime.input_prec : 0;
    if (input_prec != 0 && input_prec != 4 && input_prec != 8) { fprintf(stderr, "--input-prec must be 0, 4 or 8\n"); return 2; }
    if (*opt(argc, argv, "--input-prec", "") && input_prec) input = 1;
    if (*opt(argc, argv, "--input-mx", "") && !atoi(opt(argc, argv, "--input-mx", "0"))) input = input_prec = 0;
    if (unet_set_input_prec(input_prec)) { fprintf(stderr, "--input-prec must be 0, 4 or 8\n"); return 2; }
    unet_set_input_mx(input);
    if (sheet_task) { unet_set_input_mx(0); unet_set_input_prec(0); }
    fprintf(stderr, "prediction precision: %s checkpoint, prec=%d f16=%d act_mx4=%d act_mx8=%d input_mx=%d input_prec=%d gn_stats=%s\n", embedded ? "embedded" : "legacy", nn_get_prec(), nn_get_f16(), mx4, mx8, unet_input_converted(), unet_input_prec(), gn_stats);
    /* --gpus a,b,...: one worker process per GPU, each writes every n-th shard of the same store; the parent builds the pyramid */
    int gpus[8], ngpu = 0, part = 0;
    { const char *gl = opt(argc, argv, "--gpus", nullptr); if (gl) { char *t = strdup(gl); for (char *q = strtok(t, ","); q && ngpu < 8; q = strtok(nullptr, ",")) gpus[ngpu++] = atoi(q); free(t); } }
    if (ngpu <= 1) { if (ngpu == 1) gpu = gpus[0]; ngpu = 1; }
    /* --split z --gpus a,b: both GPUs on every window, each on one z half (see psplit_job); a and b may be one GPU (a test) */
    const int split = !strcmp(opt(argc, argv, "--split", "0"), "z");
    if (input_affine && (split || sheet_task || sweep_out || pcut >= 0)) {
        fprintf(stderr, "predict: input gain/bias is unsupported for --split, surface winding, conditioned sweeps or papyrus cutoffs\n"); return 2;
    }
    if (split && (ngpu != 2 || sheet_task || ufsm_env_on("UFSM_PRED_HOSTPATH"))) { fprintf(stderr, "--split z needs --gpus with two devices (recto / affinity checkpoints, the device path)\n"); return 2; }
    if (split) gpu = gpus[0];
    if (half && (split || sheet_task || ufsm_env_on("UFSM_PRED_HOSTPATH"))) { fprintf(stderr, "predict: half-resolution checkpoints run on one GPU per window, the device path\n"); return 2; }
    const int nwork = split ? 1 : ngpu;   /* worker processes */
    if (ffd && (split || ngpu > 1 || out_ch)) { fprintf(stderr, "predict: --fit-fields runs on one GPU and writes channel 0\n"); return 2; }
    z3 *auxz = nullptr; uint8_t *auxd = nullptr, *auxh = nullptr;   /* --aux ROOT KEY: the fifth input of 5-input checkpoints (train --aux-input) */
    {   int ai = -1; for (int k = 1; k + 2 < argc; k++) if (!strcmp(argv[k], "--aux")) ai = k;
        if (cfg.cin == 5 && !half) {
            if (ai < 0) { fprintf(stderr, "predict: %s has a fifth (aux) input: --aux ROOT KEY (the volume on this level's grid)\n", ckpt); return 2; }
            if (split || ngpu > 1) { fprintf(stderr, "predict: --aux runs on one GPU without --split\n"); return 2; }
            auxz = z3_open(store_open(argv[ai + 1]), argv[ai + 2], opt(argc, argv, "--cache", nullptr));
            if (!auxz) { fprintf(stderr, "predict: --aux: %s\n", z3_error()); return 1; }
        } else if (ai >= 0) { fprintf(stderr, "predict: --aux needs a checkpoint trained with --aux-input\n"); return 2; }
    }
    if (nwork > 1 && !box) { fprintf(stderr, "--gpus needs --box\n"); return 2; }
    if (sheet_task && ngpu>1) { fprintf(stderr,"winding export currently uses one GPU per output store\n"); return 2; }
    pid_t kids[8] = {0}; int is_child = 0;
    if (nwork > 1) {
        for (int i = 0; i < ngpu; i++) {
            pid_t c = fork();
            if (c < 0) { perror("fork"); return 1; }
            if (c == 0) { is_child = 1; gpu = gpus[i]; part = i; break; }
            kids[i] = c;
        }
        if (!is_child) {   /* parent: wait, then the pyramid and the group */
            int bad = 0; for (int i = 0; i < ngpu; i++) { int st = 0; waitpid(kids[i], &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad = 1; }
            if (bad) { fprintf(stderr, "a predict worker failed\n"); return 1; }
            double um_l = um * (1 << level);
            int64_t bn[3]; { long long v[6] = {0, 0, 0, 0, 0, 0}; const char *b = opt(argc, argv, "--box", nullptr); if (!b || sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) { fprintf(stderr, "--gpus needs --box\n"); return 1; } for (int d = 0; d < 3; d++) bn[d] = v[3 + d]; }
            if (atoi(opt(argc, argv, "--full-grid", "0"))) {   /* the pyramid of the whole grid */
                store *ps = store_open(root); z3 *pz = ps ? pyramid_open_level(ps, key, level, um, opt(argc, argv, "--cache", nullptr)) : nullptr;
                if (!pz) { fprintf(stderr, "cannot open level %d of %s: %s\n", level, key, z3_error()); return 1; }
                for (int d = 0; d < 3; d++) bn[d] = z3_meta_of(pz)->shape[d];
                z3_close(pz); store_close(ps);
            }
            char attrs[2000]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"gpus\":%d,\"ema\":%d%s%s}}", content, ckpt, ngpu, use_ema, grid_attrs, input_attrs);
            for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
            if (pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs)) return 1;
            fprintf(stderr, "wrote %s (%d GPUs)\n", out, ngpu);
            return 0;
        }
    }
    if (nn_init(gpu)) { fprintf(stderr, "cannot select GPU %d\n", gpu); return 1; }
    unet *u = unet_create(&cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "cannot load %s\n", ckpt); return 1; }
    if (ema_blend >= 0.f) unet_blend_ema(u, ema_blend);
    unet_use_ema(u, use_ema || ema_blend >= 0.f);
    const char *gn_save = opt(argc, argv, "--gn-save", nullptr), *gn_frozen = opt(argc, argv, "--gn-frozen", nullptr);
    if (gn_save && (ngpu > 1 || split)) { fprintf(stderr, "predict: --gn-save needs one GPU\n"); return 2; }
    float *gnh = nullptr;
    if (gn_frozen) {   /* header line UFSMGN{"nstats":N,...} then N float32 */
        FILE *f = fopen(gn_frozen, "rb"); char line[512]; long ns = -1;
        if (!f || !fgets(line, sizeof line, f) || strncmp(line, "UFSMGN{", 7) || !strstr(line, "\"nstats\":")) { fprintf(stderr, "predict: cannot read %s\n", gn_frozen); return 1; }
        ns = atol(strstr(line, "\"nstats\":") + 9);
        if (ns != (long)unet_gn_nstats(u)) { fprintf(stderr, "predict: %s holds %ld statistics, the network needs %zu\n", gn_frozen, ns, unet_gn_nstats(u)); return 1; }
        gnh = malloc((size_t)ns * 4);
        if (fread(gnh, 4, (size_t)ns, f) != (size_t)ns) { fprintf(stderr, "predict: %s is truncated\n", gn_frozen); return 1; }
        fclose(f);
    }
    unet *u1 = nullptr;   /* --split: the network on the second GPU */
    if (split) {
        if (nn_init(gpus[1])) { fprintf(stderr, "cannot select GPU %d\n", gpus[1]); return 1; }
        u1 = unet_create(&cfg);
        if (unet_load(u1, ckpt) < 0) { fprintf(stderr, "cannot load %s\n", ckpt); return 1; }
        if (ema_blend >= 0.f) unet_blend_ema(u1, ema_blend);
        unet_use_ema(u1, use_ema || ema_blend >= 0.f);
        nn_init(gpu);
    }
    for (int k = 0; k < 1 + split; k++) {
        unet *uk = k ? u1 : u;
        if (k) nn_init(gpus[1]);
        if (gnh) unet_gn_freeze(uk, gnh);
        if (ffd || sweep_out) unet_set_head_channels(uk, 0, cfg.cout);   /* every head: the recto channel 0 is written, the others pooled (or swept) */
        else if (!sheet_task) unet_set_head_channels(uk, hmode == 2 ? 8 * out_ch : out_ch, hmode == 2 ? 8 : 1);   /* only the written logit channel (desk widths: -24 B per window voxel) */
        /* fp16 logits (MX activations): the readers below take them as they are (nn_set_logits_h16 after each forward); the 11 heads
           of --fit-zarr were 44 B per window voxel in fp32, which pushed 640^3 windows past the 16 GB card (paged: 40x slower).
           UFSM_PRED_LOGITS32=1 keeps fp32 */
        if (!sheet_task && !half && !sweep_out && !ufsm_env_on("UFSM_PRED_LOGITS32")) unet_set_fp16_logits(uk, 1);   /* (separate from the head
                                     selection above: as the else of that chain it dropped --channel, every channel-N prediction wrote channel 0) */
        /* the network takes the uint8 CT window itself (with in-place slabs the stem builds its input per slab: no 16-bit window) */
        if (!sheet_task && !half && !ufsm_env_on("UFSM_PRED_HOSTPATH")) unet_set_ct_input(uk, 1);
        if (cfg.scale_cond && !(pcut >= 0 && pap_cond == 1)) {   /* voxel-size conditioning: the CT level's voxel size, or --scale-um to override */
            const double sum_ = atof(opt(argc, argv, "--scale-um", "0"));
            unet_set_scale(uk, sum_ > 0 ? sum_ : um * (1 << level));
        }
        if (k) nn_init(gpu);
    }
    free(gnh);
    store *s = store_open(root);
    char ak[1200];
    snprintf(ak, sizeof ak, "%s/level%d", key, level);
    z3 *ct = pyramid_open_level(s, key, level, um, cache);
    if (!ct) { fprintf(stderr, "cannot open level %d of %s: %s\n", level, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    if (cfg.view_P && pred_view_open(&cfg, root, key, level, um, cache, opt(argc, argv, "--view-level", "0"), opt(argc, argv, "--view-map", nullptr), split)) return 1;
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    { const char *b = opt(argc, argv, "--box", nullptr); long long v[6]; if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } } }
    /* --full-grid 1: the output array spans the CT level's whole grid (origin 0) and --box only selects the shards computed (rounded
       out to the shard grid), so several boxes and a later whole-volume run fill one zarr. --resume 1 skips the shards listed in
       <out>/predict-done-<level>.txt (one "z y x" line per shard, appended as each is written). --occupancy L skips shards whose
       window footprint is all zero on CT level L (> --level, e.g. 5: the scroll's air outside the mask) */
    const int fullg = atoi(opt(argc, argv, "--full-grid", "0")) != 0, resume = atoi(opt(argc, argv, "--resume", "0")) != 0;
    const int occl = atoi(opt(argc, argv, "--occupancy", "-1"));
    int64_t ash[3] = {bn[0], bn[1], bn[2]}, soff[3] = {0, 0, 0};
    if (fullg) {
        if (sheet_task || ffd || ffz || sweep_out) { fprintf(stderr, "predict: --full-grid writes plain probability volumes only (no --fit-fields / --fit-zarr / sweeps / winding)\n"); return 2; }
        for (int d = 0; d < 3; d++) {
            int64_t lo = bo[d] / shard * shard, hi = (bo[d] + bn[d] + shard - 1) / shard * shard;
            if (hi > m->shape[d]) hi = m->shape[d];
            if (hi <= lo) { fprintf(stderr, "predict: the box lies outside the CT level (%lld x %lld x %lld)\n", (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2]); return 2; }
            bo[d] = lo; bn[d] = hi - lo; ash[d] = m->shape[d]; soff[d] = lo / shard;
        }
        fprintf(stderr, "predict: full grid %lld x %lld x %lld, box rounded to shards: origin %lld,%lld,%lld size %lld,%lld,%lld\n", (long long)ash[0], (long long)ash[1], (long long)ash[2],
                (long long)bo[0], (long long)bo[1], (long long)bo[2], (long long)bn[0], (long long)bn[1], (long long)bn[2]);
    }
    if (occl >= 0 && (occl <= level || occl > 20)) { fprintf(stderr, "predict: --occupancy L needs a coarser CT level than --level (%d)\n", level); return 2; }
    const int H0 = 1 << (cfg.nlev - 1);   /* --split: halo planes of each half */
    if (split && !win_given) { W = shard + 128; halo = 64; fprintf(stderr, "predict: window %d, halo %d (split default; --window / --halo set them; window - 2 halo = k * shard computes one window per k^3 shards)\n", W, halo); }
    if (split && W % (2 * H0)) { fprintf(stderr, "--split z: the window must be a multiple of %d\n", 2 * H0); return 2; }
    if (!split && !win_given && !sheet_task && bn[0] >= shard && bn[1] >= shard && bn[2] >= shard) {   /* default window: one per k^3 shards (k shards + 2 halo,
                                          the largest k <= 4 the box and this GPU hold: in-place slab inference makes k = 2 fit 16 GB with the 16-wide net) or one per shard,
                                          halo 16 or the least multiple of 2^(levels-1) / 2 above it; else 288 / 8. Desk widths, 512 shards, laptop: 16% less GPU time per
                                          shard than 8 windows of 288, peak 9.4 GB */
        const int div = 1 << (cfg.nlev - 1), md = hdiv * div;
        const int ctin = !half && !ufsm_env_on("UFSM_PRED_HOSTPATH");   /* the device path puts the CT window in the level-0 skip buffer (unet_input_scratch2) */
        int hk = 16; while ((2 * hk) % md && hk < shard / 4) hk += 8;
        for (int k = 4; k >= 1; k--) {
            const int Wc = k * shard + 2 * hk;
            if (Wc % md || (k > 1 && (bn[0] < (int64_t)k * shard || bn[1] < (int64_t)k * shard || bn[2] < (int64_t)k * shard))) continue;
            const size_t wc3 = (size_t)Wc * Wc * Wc, need = (half ? unet_infer_bytes(u, (shape5){1, cfg.cin, Wc / hdiv, Wc / hdiv, Wc / hdiv}) + 14 * wc3 : unet_infer_bytes(u, (shape5){1, 4, Wc, Wc, Wc}))
                                                     + (ctin ? 0 : wc3) + wc3 / 8 + (size_t)shard * shard * shard + ((size_t)1 << 30);   /* (+ the CT window unless it lives in the network's level-0 buffer, its mask bits) */
            if (need <= nn_mem_free()) { W = Wc; halo = hk; break; }
        }
        if (W % md) W = (W + md - 1) / md * md;   /* (the 288 fallback, deeper nets: 320 / 8 with 7 levels) */
        fprintf(stderr, "predict: window %d, halo %d (default: one window per k^3 shards, k <= 4 the box and the GPU hold, else per shard, else 288 / 8; 288 / 16 with --fit-zarr; --window / --halo set them)\n", W, halo);
        if (box) for (int d = 0; d < 3; d++) if (boxv[d + 3] > INT64_MAX - W || boxv[d] > INT64_MAX - W - boxv[d + 3]) { fprintf(stderr, "prediction box exceeds the coordinate range\n"); return 2; }
    }
    if (ffz && !win_given && !split && W == 288 && halo == 8) halo = 16;   /* --fit-zarr fallback: a z stride (256) dividing the shard */
    if (!win_given && W % (hdiv << (cfg.nlev - 1))) W = (W + (hdiv << (cfg.nlev - 1)) - 1) / (hdiv << (cfg.nlev - 1)) * (hdiv << (cfg.nlev - 1));   /* (288 with a deeper net) */
    if (degrade && (slab || split || half || sheet_task || sweep_out || ufsm_env_on("UFSM_PRED_HOSTPATH") || W > 1024)) {
        fprintf(stderr, "predict: --aug-*: cube windows up to 1024 on one GPU's plain path\n"); return 2;
    }
    if (!slab) { Wz = W; hz = halo; }
    else if (!win_given || Wz > W || Wz % (1 << (cfg.nlev - 1)) || hz >= Wz / 2 || split || half || sheet_task || auxz || sweep_out || ufsm_env_on("UFSM_PRED_HOSTPATH")) {
        fprintf(stderr, "predict: --window-z: at most --window (given), a multiple of %d, --halo-z under half of it; one GPU's plain path (no --split, "
                        "half-resolution, winding, --aux, --cond-sweep or UFSM_PRED_HOSTPATH)\n", 1 << (cfg.nlev - 1));
        return 2;
    }
    axis ax = {0};
    if (axisf && axis_load(&ax, axisf)) { fprintf(stderr, "cannot load axis %s\n", axisf); return 1; }
    double um_l = um * (1 << level), scale = (double)(1 << level);
    if (sweep_out) {   /* conditioned inference: its own window loop and outputs (src/sweep.c) */
        if (split || ngpu > 1 || !box || !ax.n) { fprintf(stderr, "predict: --cond-sweep runs on one GPU over a --box, with --axis\n"); return 2; }
        if (!win_given) { W = 128; halo = 32; }
        sweep_cfg sw = {.u = u, .ch14 = cfg.cout - 4, .ct = ct, .bo = {bo[0], bo[1], bo[2]}, .bn = {bn[0], bn[1], bn[2]}, .um = um_l, .scale = scale,
                        .ax = &ax, .axis_path = axisf, .axis_flag = (int)radial_sign, .W = W, .halo = halo,
                        .h16 = nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8"), .nthreads = nthreads,
                        .dir = atoi(opt(argc, argv, "--cond-dir", "1")), .reverse = atoi(opt(argc, argv, "--cond-reverse", "0")), .raw = atoi(opt(argc, argv, "--cond-raw", "0")), .out = sweep_out, .ckpt = ckpt};
        fprintf(stderr, "predict: conditioned sweep, window %d, halo %d%s -> %s\n", W, halo, sw.reverse ? ", reversed order" : "", sweep_out);
        return cond_sweep(&sw);
    }
    /* output array covers the box, with origin recorded in attributes */
    char lv[32], ldir[1400], attrs[1024];
    z3w_level_name(um_l, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"step\":%d,\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d,\"window_z\":%d,\"halo_z\":%d,\"ema\":%d%s%s}}",
             content, ckpt, step, root, ak, (long long)(fullg ? 0 : bo[0]), (long long)(fullg ? 0 : bo[1]), (long long)(fullg ? 0 : bo[2]), W, halo, Wz, hz, use_ema, grid_attrs, input_attrs);
    z3w *w = z3w_create(ldir, ash, shard, q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    int stride = W - 2 * halo;
    const int strd[3] = {Wz - 2 * hz, stride, stride}, hal[3] = {hz, halo, halo}, wn[3] = {Wz, W, W};   /* per axis (slabs: z differs) */
    size_t w3 = (size_t)Wz * W * W;
    float *sheet_window=nullptr,*sheet_shard=nullptr,*sheet_rows=nullptr;
    if (sheet_task) {
        char path[1400]; snprintf(path,sizeof path,"%s/winding",out); if (mkdir(path,0755) && errno!=EEXIST) return 1;
        sheet_window=malloc(w3*4); sheet_shard=malloc((size_t)shard*shard*shard*4); sheet_rows=malloc(4*(size_t)W*sizeof(float));
        if (!sheet_window || !sheet_shard || !sheet_rows) return 1;
        snprintf(path,sizeof path,"%s/winding/manifest.json",out); FILE *f=fopen(path,"w"); if (!f) return 1;
        fprintf(f,"{\"version\":1,\"task\":\"surface_winding\",\"dtype\":\"<f4\",\"codec\":\"zstd\",\"units\":\"turns\",\"shape\":[%lld,%lld,%lld],\"origin_zyx\":[%lld,%lld,%lld],\"shard\":%d,\"validity\":\"finite and surface probability >= 0.1\",\"geometry_sha256\":\"%s\",\"reference_sha256\":\"%s\"}\n",(long long)bn[0],(long long)bn[1],(long long)bn[2],(long long)bo[0],(long long)bo[1],(long long)bo[2],shard,geometry_sha,reference_sha);
        if (fclose(f)) return 1;
    }
    writer_t WR = {0}; WR.w = w; WR.nthreads = nthreads; WR.bytes = (size_t)shard * shard * shard;
    WR.skip = atoi(opt(argc, argv, "--no-out", "0")) != 0;   /* --no-out 1: the prediction volume's shards are not written (only its metadata; e.g. with --fit-zarr) */
    for (int d = 0; d < 3; d++) WR.off[d] = soff[d];
    WR.donefd = -1;
    char donep[1500]; snprintf(donep, sizeof donep, "%s/predict-done-%s.txt", out, lv);
    if (resume && !WR.skip && (WR.donefd = open(donep, O_WRONLY | O_CREAT | O_APPEND, 0644)) < 0) { fprintf(stderr, "predict: cannot open %s: %s\n", donep, strerror(errno)); return 1; }
    /* default path (UFSM_PRED_HOSTPATH=1: the previous host path, for A/B): pinned reader / shard buffers, window statistics
       on the device, the interior probabilities placed into a device shard buffer that is downloaded once per shard */
    const int gpath = !ufsm_env_on("UFSM_PRED_HOSTPATH");
    WR.nb = 2;   /* (the ring grows below once the window grid is known: one window per k^3 shards) */
    WR.buf[0] = gpath ? nn_host_alloc(WR.bytes) : malloc(WR.bytes); WR.buf[1] = gpath ? nn_host_alloc(WR.bytes) : malloc(WR.bytes);
    uint8_t *dsh = gpath && !split ? nn_malloc(WR.bytes) : nullptr; void *stsc = nn_malloc(64); pthread_mutex_init(&WR.mu, nullptr); pthread_cond_init(&WR.cv, nullptr);
    WR.gpu = gpu; WR.async = gpath && !split && !(getenv("UFSM_PRED_SYNC_D2H") && atoi(getenv("UFSM_PRED_SYNC_D2H")));
    if (WR.async) for (int k = 0; k < WBUF_MAX; k++) { WR.evs[k] = nn_event_create(); WR.evd[k] = nn_event_create(); }
    pthread_t wth; pthread_create(&wth, nullptr, writer_main, &WR);
    int sb = 0; uint8_t *sbuf = WR.buf[0];
    /* the window goes up as uint8; the input channels are built on the device in the network's storage type */
    const int h16 = nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8");
    uint8_t *pu = gpath ? nullptr : malloc(w3), *pud = gpath ? nullptr : nn_malloc(w3);
    float *dyo = malloc(2 * (size_t)W * sizeof(float)), *dxo = dyo + W, *dyd = nn_malloc(2 * (size_t)W * sizeof(float)), *dxd = dyd + W;
    const int Dl = split ? W / 2 + H0 : Wz;   /* planes per GPU */
    shape5 xs = {1, !half && (cfg.cin == 5 || cfg.cin == 7) ? cfg.cin : 4, Dl, W, W};   /* (7: a train --cond net, its condition inputs zero here) */
    const shape5 xn = half ? (shape5){1, cfg.cin, W / hdiv, W / hdiv, W / hdiv} : xs;   /* the network's input */
    void *x2 = nullptr, *lrt = nullptr; float *lgf = nullptr;   /* half: the network's input, the intermediate grid (div 4), the full-resolution logits */
    /* the 16-bit input in the network's shared a1 scratch when it has one (free until the stem conv, after the input conversion) */
    unet_build(u, xn, 0);
    const int ct_stem = unet_ct_stem(u);   /* unet_forward_ct: no 16-bit input */
    uint8_t *ctd = nullptr, *ctcp = nullptr; uint32_t *ctm = nullptr, *ctmi = nullptr; void *xd = nullptr; int ct_shared = 0, xd_shared = 0;
    pjob_t J = {0};
    split_ctx *sctx = nullptr;
    if (split) {   /* per GPU: the network on its half, the CT half in its scratch, the mask bits, a device shard */
        const size_t l3 = (size_t)Dl * W * W;
        for (int k = 0; k < 2; k++) {
            unet *uk = k ? u1 : u; pside_t *S = &J.s[k];
            nn_init(gpus[k]);
            unet_set_split(uk, k, H0, split_halo); unet_set_split_async(uk, split_halo_begin, split_halo_end);
            unet_build(uk, xs, 0);
            S->u = uk; S->ctd = unet_input_scratch2(uk, l3);
            if (!unet_ct_stem(uk) || !S->ctd) { fprintf(stderr, "--split z needs the CT-stem build (UFSM_INFER_SLAB > 0, an MX-fp8 or MX-fp4 stem input)\n"); return 2; }
            S->ctm = nn_malloc((l3 + 31) / 32 * 4); S->ctmi = iso > 1 ? nn_malloc((l3 + 31) / 32 * 4) : S->ctm; S->dsh = nn_malloc(WR.bytes); S->stsc = nn_malloc(64); S->red = nn_malloc(4 * sizeof(double));
            S->dyd = nn_malloc(2 * (size_t)Dl * sizeof(float)); S->rows = malloc(2 * (size_t)Dl * sizeof(float)); S->own = calloc((size_t)shard, 1);
            if (!S->ctm || !S->dsh || !S->stsc || !S->red || !S->dyd) { fprintf(stderr, "predict: out of device memory on GPU %d\n", gpus[k]); return 1; }
            fprintf(stderr, "predict: GPU %d holds window planes [%d, %d) of %d (%.1f B per voxel held, %.2f GB)\n", gpus[k], k ? W / 2 - H0 : 0, (k ? W / 2 - H0 : 0) + Dl, W, (double)unet_activation_bytes(uk) / l3, unet_activation_bytes(uk) / 1e9);
        }
        nn_init(gpu);
        sctx = split_create(gpus[0], gpus[1]);
    } else {
        /* the uint8 CT window in the network's level-0 skip buffer (free until the forward's input conversion); the output mask then
           goes through a bit plane (the buffer does not survive the forward) */
        ctd = gpath && !sheet_task ? unet_input_scratch2(u, w3) : nullptr;
        ct_shared = ctd != nullptr;
        if (!ctd) ctd = nn_malloc(w3);
        if (ffct && ct_shared && !(ctcp = nn_malloc(w3))) { fprintf(stderr, "predict: no device memory for the --fit-ct CT copy\n"); return 1; }
        ctm = ct_shared ? nn_malloc((w3 + 31) / 32 * 4) : nullptr;
        ctmi = ct_shared && iso > 1 ? nn_malloc((w3 + 31) / 32 * 4) : ctm;   /* --iso: the outputs' own mask (the fit keeps ctm) */
        xd = ct_stem || half || auxz || cfg.cin == 7 ? nullptr : h16 && !sheet_task ? unet_input_scratch(u, 4 * w3 * 2) : nullptr;
        xd_shared = xd != nullptr;
        if (!xd && !ct_stem) xd = nn_malloc((size_t)xs.c * w3 * (h16 ? 2 : 4));
        if (auxz) { auxd = nn_malloc(w3); auxh = malloc(w3); }
        if (half) { x2 = nn_malloc(shape_numel(xn) * (h16 ? 2 : 4)); lgf = nn_malloc(w3 * 4); if (hdiv > 2) lrt = nn_malloc(w3 / 8 * 16); }
        if (ct_stem) fprintf(stderr, "predict: in-place slab inference from the CT window (%.1f B per window voxel)\n", (double)unet_activation_bytes(u) / w3);
        if (tta > 1 && (!ct_stem || !ct_shared || degrade || auxz || cfg.cin != 4)) { fprintf(stderr, "predict: --tta needs the CT-stem device path (no --aug-*, --aux, extra inputs)\n"); return 2; }
    }
    int64_t ns[3];
    for (int d = 0; d < 3; d++) ns[d] = (bn[d] + shard - 1) / shard;
    /* --resume / --occupancy: the box's shards not to compute (1 = already written, 2 = air: the outputs are masked to CT > 0, so a shard
       whose own region is zero on the coarse level, one coarse voxel of margin, predicts nothing) */
    uint8_t *sskip = nullptr; long nresumed = 0, nocc = 0;
    uint8_t *occb = nullptr; int64_t occo[3] = {0, 0, 0}, occn[3] = {0, 0, 0}; int occf = 1;   /* --occupancy: the coarse level over the box (tile tests below) */
    if (resume || occl >= 0) {
        sskip = calloc((size_t)(ns[0] * ns[1] * ns[2]), 1);
        if (!sskip) { fprintf(stderr, "predict: out of memory\n"); return 1; }
        FILE *df = resume ? fopen(donep, "r") : nullptr;
        long long gz, gy, gx;
        if (df) {
            while (fscanf(df, "%lld %lld %lld", &gz, &gy, &gx) == 3) {
                const int64_t lz = gz - soff[0], ly = gy - soff[1], lx = gx - soff[2];
                if (lz < 0 || ly < 0 || lx < 0 || lz >= ns[0] || ly >= ns[1] || lx >= ns[2]) continue;
                uint8_t *e = &sskip[(lz * ns[1] + ly) * ns[2] + lx]; if (!*e) { *e = 1; nresumed++; }
            }
            fclose(df);
        }
        if (occl >= 0) {
            z3 *oc = pyramid_open_level(s, key, occl, um, cache);
            if (!oc) { fprintf(stderr, "predict: --occupancy: cannot open CT level %d: %s\n", occl, z3_error()); return 1; }
            const z3_meta *om = z3_meta_of(oc); const int f = 1 << (occl - level);
            int64_t o0[3], n0[3];
            for (int d = 0; d < 3; d++) {
                int64_t lo = bo[d] / f - 1, hi = (bo[d] + bn[d] + f - 1) / f + 1;
                if (lo < 0) lo = 0;
                if (hi > om->shape[d]) hi = om->shape[d];
                o0[d] = lo; n0[d] = hi > lo ? hi - lo : 0;
            }
            uint8_t *ob = n0[0] && n0[1] && n0[2] ? malloc((size_t)(n0[0] * n0[1] * n0[2])) : nullptr;
            if (n0[0] && n0[1] && n0[2] && (!ob || z3_read(oc, o0, n0, ob, nthreads))) { fprintf(stderr, "predict: --occupancy: cannot read CT level %d: %s\n", occl, z3_error()); return 1; }
            for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
                uint8_t *e = &sskip[(sz * ns[1] + sy) * ns[2] + sx];
                if (*e) continue;
                const int64_t ss[3] = {sz, sy, sx}; int64_t a[3], b[3]; int hit = 0;
                for (int d = 0; d < 3; d++) {
                    const int64_t g0 = bo[d] + ss[d] * shard, g1 = g0 + shard < bo[d] + bn[d] ? g0 + shard : bo[d] + bn[d];
                    a[d] = g0 / f - 1 - o0[d]; b[d] = (g1 + f - 1) / f + 1 - o0[d];
                    if (a[d] < 0) a[d] = 0;
                    if (b[d] > n0[d]) b[d] = n0[d];
                }
                for (int64_t z = a[0]; z < b[0] && !hit; z++) for (int64_t y = a[1]; y < b[1] && !hit; y++) {
                    const uint8_t *r = ob + ((size_t)z * n0[1] + y) * n0[2];
                    for (int64_t x = a[2]; x < b[2]; x++) if (r[x]) { hit = 1; break; }
                }
                if (!hit) { *e = 2; nocc++; }
            }
            occb = ob; occf = f; for (int d = 0; d < 3; d++) { occo[d] = o0[d]; occn[d] = n0[d]; }
            z3_close(oc);
        }
        fprintf(stderr, "predict: %lld shards in the box: %ld already written (--resume), %ld air on CT level %d (--occupancy), %lld to compute\n",
                (long long)(ns[0] * ns[1] * ns[2]), nresumed, nocc, occl, (long long)(ns[0] * ns[1] * ns[2]) - nresumed - nocc);
    }
    /* interior = k whole shards (k >= 2, the tile grid on the shard grid): one window per k^3 shards, each shard placed from it and
       written (INFERENCE_MEMORY_PLAN.md, memory into speed); otherwise one or more windows per shard, clipped to it */
    int kss = !slab && !sheet_task && gpath && stride % shard == 0 && stride / shard >= 2 ? stride / shard : 1;
    for (int d = 0; d < 3 && kss > 1; d++) { int64_t ph = grid ? (bo[d] - grid_origin[d]) % stride : 0; if (ph < 0) ph += stride; if (ph) kss = 1; }
    if (fea) {   /* --feather: the window grid on the shard grid, interior k shards (k >= 1), one GPU path */
        int bad = fea < 0 || fea > halo || fea > hz || fea > 1024 || slab || sheet_task || !gpath || split || ffd || half || stride % shard || sweep_out;
        for (int d = 0; d < 3 && !bad; d++) { int64_t ph = grid ? (bo[d] - grid_origin[d]) % stride : 0; if (ph < 0) ph += stride; if (ph) bad = 1; }
        if (bad) { fprintf(stderr, "predict: --feather B needs 0 < B <= halo (<= 1024), window - 2 halo = k * shard with the box on that grid, the plain device path "
                               "(no --split, --fit-*, --window-z, half-resolution or winding outputs)\n"); return 2; }
        kss = stride / shard;
    }
    const int sup = kss > 1 || fea > 0;   /* one window per k^3 shards (k = 1 with --feather), placed and written per window */
    int64_t nss[3]; for (int d = 0; d < 3; d++) nss[d] = (ns[d] + kss - 1) / kss;
    float *ffacc = nullptr, *ffacc4 = nullptr; int ffbox[3], ffF[3]; size_t ffn = 0;
    if (ffu2) {   /* the relative planes need the far voxel inside the window: offset <= the halos */
        if (ffw4d > halo) ffw4d = halo;
        if (ffw4d > hz) ffw4d = hz;
        if (ffw4d < 2) { fprintf(stderr, "predict: the winding mod 4 planes need a halo of >= 2 voxels (--fit-w4-d %d): not written\n", ffw4d); ffw4d = 0; }
        else fprintf(stderr, "predict: task unwrap2 checkpoint: fit fields from channels 0-1 phase, 2-7 normal, 8 density; winding mod 4 planes dz dy dx dq over %d voxels\n", ffw4d);
    }
    if (ffu2) {   /* (no recto plane) */
        static const char *const w4n[4] = {"dz", "dy", "dx", "dq"};
        ffz_np = 10; if (ffw4d) for (int k = 0; k < 4; k++) ffz_names[ffz_np++] = w4n[k];
    }
    ffz_nb = ffz_np;
    if (ffct) { ffz_names[ffz_np++] = "ct"; ffz_names[ffz_np++] = "cs"; }
    z3w *ffzw[FFZ_MAXP] = {0}; uint8_t *ffsl[FFZ_MAXP] = {0}; const int FFS = 128; int FZ = FFS; int64_t ffslab = 0; uint8_t *ff8d = nullptr, *ff8h = nullptr; size_t ffwn = 0; int ffwc = 0, ffzc = 1;
    static ffw_t ffw; pthread_t ffw_th; int ffw_on = 0, ffrow = 0, ffrk = 0; int64_t ffprow = -1;   /* row flushes: (super-)shard cells, the last row */
    if (ffd) {
        for (int d = 0; d < 3; d++) { ffbox[d] = (int)bn[d]; ffF[d] = (int)((bn[d] + ffh - 1) / ffh); }
        if (!ffzm) {   /* the whole box on the device */
            ffn = (size_t)ffF[0] * ffF[1] * ffF[2]; ffacc = nn_malloc(ffna * ffn * 4);
            if (!ffacc) { fprintf(stderr, "predict: no device memory for the fit fields\n"); return 1; }
            nn_zero(ffacc, ffna * ffn * 4);
            if (ffw4d) { ffacc4 = nn_malloc(9 * ffn * 4); if (!ffacc4) { fprintf(stderr, "predict: no device memory for the fit fields\n"); return 1; } nn_zero(ffacc4, 9 * ffn * 4); }
        } else {   /* per window: an accumulator of its interior cells; the field slab of FZ cells in z on the host */
            /* the field slab (FZ cells in z, a multiple of the field shard FFS) is flushed once the windows leave it: a window's
               interior must not straddle two (288 / 8, stride 272, wrote past the slab and crashed): per-shard windows have z strides
               dividing the shard, super-shard windows (stride k shard: one window writes k^3 shards, e.g. --window 384 --halo 64 with
               --shard 128) interiors of k shards, so FZ h is a multiple of the shard (or the super-shard) */
            const int zsh = kss > 1 ? kss * shard : shard;
            while ((FZ * ffh) % zsh && FZ < 4096) FZ += FFS;
            if (stride % ffh || strd[0] % ffh || shard % ffh || (FZ * ffh) % zsh || (kss == 1 && shard % strd[0])) {
                fprintf(stderr, "predict: --fit-zarr needs the window strides (window - 2 halo; z: %d) divisible by --fit-h and the shard by --fit-h; "
                                "per-shard windows a z stride dividing the shard (e.g. --window 288 --halo 16 or 544 / 16 with --shard 512), "
                                "else a stride of k shards (e.g. --window 384 --halo 64 --shard 128)\n", strd[0]);
                return 2;
            }
            for (int d = 0; d < 3; d++) { const int64_t ph = grid ? (bo[d] - grid_origin[d]) % ffh : 0; if (ph) { fprintf(stderr, "predict: --fit-zarr: the box must be aligned to --fit-h on the tile grid\n"); return 2; } }
            char at[1400]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h)\",\"h\":%d,\"origin_zyx\":[%lld,%lld,%lld],\"checkpoint\":\"%s\",\"volcomp_q\":%g,\"volcomp_q_phase\":%g}}", ffh, (long long)bo[0], (long long)bo[1], (long long)bo[2], ckpt, ffq, ffqp);
            /* rows of field shards are written as soon as the windows leave their row of (super-)shards (a window writes cells of its own
               (super-)shard only, tiles run in (super-)shard z-y-x order): only the last row's compression is left after the loop (was
               the whole last slab: 8 of 24 s for a 256 x 2048^2 box). One worker, a (super-)shard of whole field shards; UFSM_FIT_ROWFLUSH=0.
               With them the slab is a ring of 3 rows when a row is the slab's height (FZ = ffrk); UFSM_FIT_RING=0: the cross-section */
            ffrk = (kss > 1 ? kss * shard : shard) / ffh;
            ffrow = nwork == 1 && !split && ffrk % FFS == 0 && !(getenv("UFSM_FIT_ROWFLUSH") && !atoi(getenv("UFSM_FIT_ROWFLUSH")));
            const int ring = ffrow && FZ == ffrk && 3 * (int64_t)ffrk < ffF[1] && !(getenv("UFSM_FIT_RING") && !atoi(getenv("UFSM_FIT_RING")));
            ffw.rg = (ffring){.HZ = ffF[0] < FZ ? ffF[0] : FZ, .YR = ring ? ffrk : ffF[1], .NR = ring ? 3 : 1, .mu = &ffw.mu, .cv = &ffw.cv};
            ffw.rg.nRR = (ffF[1] + ffw.rg.YR - 1) / ffw.rg.YR; ffw.rg.own = malloc(sizeof(int64_t) * ffw.rg.NR);
            if (!ffw.rg.own) { fprintf(stderr, "predict: out of memory\n"); return 1; }
            for (int s = 0; s < ffw.rg.NR; s++) ffw.rg.own[s] = -1;
            if (ring) fprintf(stderr, "predict: fit field slab: a ring of 3 rows of %d cells, %.1f GB (the cross-section: %.1f GB)\n", ffrk,
                              (double)ffz_np * ffw.rg.HZ * 3 * ffrk * ffF[2] / 1e9, (double)ffz_np * ffw.rg.HZ * ffF[1] * ffF[2] / 1e9);
            const int64_t fsh[3] = {ffF[0], ffF[1], ffF[2]};
            for (int k = 0; k < ffz_np; k++) {
                char d[1400]; snprintf(d, sizeof d, "%s/%s", ffz, ffz_names[k]);
                const size_t held = (size_t)ffw.rg.NR * ffw.rg.HZ * ffw.rg.YR * ffF[2];
                ffzw[k] = z3w_create(d, fsh, FFS, k < 2 ? ffqp : ffq, ffz_fill[k], at);
                ffsl[k] = malloc(held);
                if (!ffzw[k] || !ffsl[k]) { fprintf(stderr, "predict: --fit-zarr: %s\n", z3w_error()); return 1; }
                memset(ffsl[k], ffz_fill[k], held);
            }
            /* a window's cells in z chunks of ffzc planes: 70 device bytes per cell (12 fp32 sums + two 11-byte staging buffers), so at --fit-h 1 a
               whole 512^3 interior took 7.8 GB and pushed 640^3 windows past the 16 GB card (paged); <= ~0.5 GB now */
            ffwc = stride / ffh + 1;
            { const size_t pc = (size_t)ffwc * ffwc, nzc = (size_t)(strd[0] / ffh + 1), cap = ((size_t)512 << 20) / ((48 + (ffw4d ? 36 : 0) + 2 * FFZ_MAXP) * pc);
              ffzc = (int)(nzc < (cap ? cap : 1) ? nzc : (cap ? cap : 1));
              if (tta > 1) ffzc = (int)nzc;   /* every pass pools into one accumulator of the whole interior */
              ffwn = (size_t)ffzc * pc; }
            ffacc = nn_malloc(ffna * ffwn * 4); ff8d = nn_malloc(2 * (size_t)FFZ_MAXP * ffwn); ff8h = nn_host_alloc(2 * (size_t)FFZ_MAXP * ffwn);
            if (ffw4d && !(ffacc4 = nn_malloc(9 * ffwn * 4))) { fprintf(stderr, "predict: no memory for the fit-field window accumulator\n"); return 1; }   /* finalized on the device: 11 B per cell back,
                                                                                                                  two staging buffers (the fit writer thread) */
            if (!ffacc || !ff8d || !ff8h) { fprintf(stderr, "predict: no memory for the fit-field window accumulator\n"); return 1; }
            ffw.zw = ffzw; ffw.sl = ffsl; ffw.F = ffF; ffw.FS = FFS; ffw.FZ = FZ; ffw.gpu = gpu;
            ffw.done = calloc((size_t)(FZ / FFS) * ((ffF[1] + FFS - 1) / FFS), 1);
            if (!ffw.done) { fprintf(stderr, "predict: out of memory\n"); return 1; }
            for (int b = 0; b < 2; b++) { ffw.h[b] = ff8h + (size_t)b * FFZ_MAXP * ffwn; ffw.evs[b] = nn_event_create(); ffw.evd[b] = nn_event_create(); }
            pthread_mutex_init(&ffw.mu, nullptr); pthread_cond_init(&ffw.cv, nullptr); atomic_store(&ffw.fail, 0);
            { long nc = ufsm_ncpu(); ffw.rnt = nc > 2 ? (int)(nc / 2) : 1; }
            if (pthread_create(&ffw_th, nullptr, ffw_main, &ffw) || pthread_create(&ffw.rth, nullptr, ffw_rows, &ffw)) { fprintf(stderr, "predict: cannot start the fit writer\n"); return 1; }
            ffw_on = 1;
        }
    }
    if (kss > 1) fprintf(stderr, "predict: one window per %d^3 shards (interior %d)\n", kss, stride);
    {   /* the writer ring: one window per k^3 shards hands over k^3 shards at once (2 k^3 buffers: the next window's shards while these
           are compressed); UFSM_PRED_WBUF=N overrides (2 = the previous double buffer) */
        int nb = sup && gpath && !split ? 2 * kss * kss * kss : 2;
        if (getenv("UFSM_PRED_WBUF")) nb = atoi(getenv("UFSM_PRED_WBUF"));
        if (nb < 2) nb = 2;
        if (nb > WBUF_MAX) nb = WBUF_MAX;
        for (int k = 2; k < nb; k++) {
            WR.buf[k] = gpath ? nn_host_alloc_try(WR.bytes) : malloc(WR.bytes);
            if (!WR.buf[k]) { nb = k; break; }
        }
        pthread_mutex_lock(&WR.mu); WR.nb = nb; pthread_mutex_unlock(&WR.mu);
        if (nb > 2) fprintf(stderr, "predict: %d shard buffers (%.1f GB pinned)\n", nb, nb * (double)WR.bytes / 1e9);
    }
    double t0 = now(); long ntiles = 0, nskip = 0;
    /* enumerate the tiles in shard order, then let the reader run ahead */
    reader_t rd = {0}; rd.ct = ct; rd.bo = bo; rd.W = W; rd.Wz = Wz; rd.nthreads = nthreads;
    long cap = 0;
    long nwq = 0;   /* the non-air (super-)shards dealt round-robin to the GPU workers. Air (--occupancy, 2) is the same in every worker,
                       but each worker reads the --resume list when it starts, so a shard another worker wrote meanwhile is "written" (1)
                       in a late one only: the deal counts those, then the owner skips them (dealing by the late view left shards to
                       no worker) */
    int64_t fz0 = 0, fz1 = nss[0];   /* --feather with --gpus: this worker's contiguous window layers (and the layer before them, unwritten) */
    if (fea && nwork > 1) { fz0 = nss[0] * part / ngpu; fz1 = nss[0] * (part + 1) / ngpu; }
    uint8_t *fneed = nullptr;   /* --feather: window to write (1), or computed only for the spills of a later one (2) */
    if (fea) {
        fneed = calloc((size_t)(nss[0] * nss[1] * nss[2]), 1);
        if (!fneed) { fprintf(stderr, "predict: out of memory\n"); return 1; }
        for (int64_t sz = fz0; sz < fz1; sz++) for (int64_t sy = 0; sy < nss[1]; sy++) for (int64_t sx = 0; sx < nss[2]; sx++) {
            int all = 1;
            for (int64_t a = sz * kss; a < (sz + 1) * kss && a < ns[0]; a++) for (int64_t b = sy * kss; b < (sy + 1) * kss && b < ns[1]; b++)
                for (int64_t c = sx * kss; c < (sx + 1) * kss && c < ns[2]; c++) all &= sskip && sskip[(a * ns[1] + b) * ns[2] + c] != 0;
            if (!all) fneed[(sz * nss[1] + sy) * nss[2] + sx] = 1;
        }
        for (int64_t sz = fz0 > 0 ? fz0 - 1 : 0; sz < fz1; sz++) for (int64_t sy = 0; sy < nss[1]; sy++) for (int64_t sx = 0; sx < nss[2]; sx++) {
            uint8_t *e = &fneed[(sz * nss[1] + sy) * nss[2] + sx]; int air = 1;
            if (*e) continue;
            for (int64_t a = sz * kss; a < (sz + 1) * kss && a < ns[0]; a++) for (int64_t b = sy * kss; b < (sy + 1) * kss && b < ns[1]; b++)
                for (int64_t c = sx * kss; c < (sx + 1) * kss && c < ns[2]; c++) air &= sskip && sskip[(a * ns[1] + b) * ns[2] + c] == 2;
            if (air) continue;   /* (air: absent, its neighbours renormalize) */
            for (int d = 1; d < 8 && !*e; d++) {   /* a later neighbour to write needs its spill */
                const int64_t qa = sz + (d >> 2 & 1), qb = sy + (d >> 1 & 1), qc = sx + (d & 1);
                if (qa < nss[0] && qb < nss[1] && qc < nss[2] && fneed[(qa * nss[1] + qb) * nss[2] + qc] == 1) *e = 2;
            }
        }
        long nw = 0, nsp = 0; for (int64_t k = 0; k < nss[0] * nss[1] * nss[2]; k++) { nw += fneed[k] == 1; nsp += fneed[k] == 2; }
        fprintf(stderr, "predict: --feather %d: %ld windows (interior %d) to write, %ld more for their spills only\n", fea, nw, stride, nsp);
        for (int64_t sz = 0; sz < nss[0]; sz++) for (int64_t sy = 0; sy < nss[1]; sy++) for (int64_t sx = 0; sx < nss[2]; sx++) {
            const uint8_t e = fneed[(sz * nss[1] + sy) * nss[2] + sx];
            if (!e) continue;
            if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
            rd.tiles[rd.ntiles++] = (tile_t){sz * stride - halo, sy * stride - halo, sx * stride - halo, (sz * nss[1] + sy) * nss[2] + sx, e == 2};
        }
    } else
    if (kss > 1) for (int64_t sz = 0; sz < nss[0]; sz++) for (int64_t sy = 0; sy < nss[1]; sy++) for (int64_t sx = 0; sx < nss[2]; sx++) {   /* super-shards */
        int all = 1, air = 1;   /* skipped when every shard of it is; dealt unless every shard is air */
        if (sskip) {
            for (int64_t a = sz * kss; a < (sz + 1) * kss && a < ns[0]; a++) for (int64_t b = sy * kss; b < (sy + 1) * kss && b < ns[1]; b++)
                for (int64_t c = sx * kss; c < (sx + 1) * kss && c < ns[2]; c++) { const uint8_t e = sskip[(a * ns[1] + b) * ns[2] + c]; all &= e != 0; air &= e == 2; }
        } else all = air = 0;
        if (air) continue;
        if (nwork > 1 && (nwq++ % ngpu) != part) continue;
        if (all) continue;
        if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
        rd.tiles[rd.ntiles++] = (tile_t){sz * stride - halo, sy * stride - halo, sx * stride - halo, (sz * nss[1] + sy) * nss[2] + sx, 0};
    }
    /* (tile tests: not with --gn-save, whose statistics count every non-air window, nor --feather, whose neighbours need a skipped window's spill;
       UFSM_OCC_TILES=0 off) */
    const int occt = occb && !gn_save && !fea && !sheet_task && !split && (!getenv("UFSM_OCC_TILES") || ufsm_env_on("UFSM_OCC_TILES")); long nocct = 0;
    if (!sup) for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        int64_t so[3] = {sz * shard, sy * shard, sx * shard}, se[3], first[3];
        const uint8_t sk = sskip ? sskip[(sz * ns[1] + sy) * ns[2] + sx] : 0;
        if (sk == 2) continue;   /* --occupancy: air */
        if (nwork > 1 && (nwq++ % ngpu) != part) continue;   /* another worker's shard */
        if (sk) continue;   /* --resume: written (see nwq) */
        for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
        for (int d = 0; d < 3; d++) {
            int64_t phase = grid ? (bo[d] + so[d] - grid_origin[d]) % strd[d] : 0;
            if (phase < 0) phase += strd[d];
            /* Read the complete globally anchored window, including CT outside
               the requested box. Clip only its output into the current shard. */
            first[d] = so[d] - phase - hal[d];
        }
        for (int64_t tz = first[0]; tz + hz < se[0]; tz += strd[0])
        for (int64_t ty = first[1]; ty + halo < se[1]; ty += stride)
        for (int64_t tx = first[2]; tx + halo < se[2]; tx += stride) {
            if (occt) {   /* --occupancy: a window whose interior (in this shard) is air on the coarse level writes nothing (outputs and
                             fit fields count CT > 0 interior voxels only): not read, not run */
                const int64_t tw[3] = {tz, ty, tx}; int64_t g0[3], g1[3];
                for (int d = 0; d < 3; d++) {
                    const int64_t lo = tw[d] + hal[d], hi = tw[d] + wn[d] - hal[d];
                    g0[d] = bo[d] + (lo > so[d] ? lo : so[d]); g1[d] = bo[d] + (hi < se[d] ? hi : se[d]);
                }
                if (!occ_any(occb, occo, occn, occf, g0, g1)) { nocct++; continue; }
            }
            if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
            rd.tiles[rd.ntiles++] = (tile_t){tz, ty, tx, (sz * ns[1] + sy) * ns[2] + sx, 0};
        }
    }
    if (occt) fprintf(stderr, "predict: %ld windows to run, %ld more skipped (interior air on CT level %d, --occupancy)\n", rd.ntiles, nocct, occl);
    free(occb); occb = nullptr;
    for (int k = 0; k < NSLOT; k++) rd.buf[k] = gpath ? nn_host_alloc(w3) : malloc(w3);
    rd.nostats = gpath;
    atomic_store(&rd.filled, 0); atomic_store(&rd.consumed, 0); atomic_store(&rd.failed, 0);
    pthread_mutex_init(&rd.mu, nullptr); pthread_cond_init(&rd.cv, nullptr);
    pthread_t rth; pthread_create(&rth, nullptr, reader_main, &rd);
    feather_t FE = {0}; const int FEN = stride + fea; uint8_t *fext = nullptr, *fhost = nullptr;   /* --feather: E on the device and pinned on the host */
    static fe_worker FW; long fe_n = 0;
    if (fea) {
        const size_t e3 = (size_t)FEN * FEN * FEN;
        /* E on the device: in the CT window's buffer (the level-0 skip scratch, dead from the end of the forward until the next window's
           upload) when it holds E and the logits do not overlap it (checked per window), else its own buffer (1.2 GB at interior 1024) */
        if (feather_init(&FE, stride, fea, nss) || (!(ct_shared && w3 >= e3) && !(fext = nn_malloc_try(e3))) || !(fhost = nn_host_alloc_try(2 * e3))) { fprintf(stderr, "predict: --feather: out of memory (%.2f GB device + 2x pinned host)\n", e3 / 1e9); return 1; }
        FW = (fe_worker){.F = &FE, .WR = &WR, .sb = sb, .kss = kss, .shard = shard, .ns = ns, .bn = bn, .fh = {fhost, fhost + e3}};
        pthread_mutex_init(&FW.mu, nullptr); pthread_cond_init(&FW.cv, nullptr);
        if (pthread_create(&FW.th, nullptr, fe_main, &FW)) { fprintf(stderr, "predict: cannot start the feather thread\n"); return 1; }
    }
    int64_t cur_shard = -1, so[3] = {0, 0, 0}, se[3] = {0, 0, 0}, sz = 0, sy = 0, sx = 0;
    int any = 0;
    const int pprof = ufsm_env_on("UFSM_PRED_PROF");
    double p_wait = 0, p_h2d = 0, p_gpu = 0, p_copy = 0, p_wr = 0, p_ffw = 0, pt;
    /* the next window's CT uploaded on the copy stream during this window's forward (into ctpre, w3 bytes; at the window's start one
       device copy into ctd): the synchronous upload left the GPU idle ~9 ms per 384 x 640^2 window. Not with --aug-* (degrade rewrites
       the host window first); UFSM_PRED_CTPRE=0 off. long pre_i: the window whose CT ctpre holds (or is receiving) */
    uint8_t *ctpre = nullptr; void *ev_pre = nullptr, *ev_preu = nullptr; long pre_i = -1;
    if (gpath && !degrade && !split && (!getenv("UFSM_PRED_CTPRE") || ufsm_env_on("UFSM_PRED_CTPRE")) && (ctpre = nn_malloc_try(w3))) {
        ev_pre = nn_event_create(); ev_preu = nn_event_create();
    }
    if (split) {   /* both GPUs on every window */
        J.rd = &rd; J.WR = &WR; J.tiles = rd.tiles; J.ntiles = rd.ntiles;
        J.W = W; J.halo = halo; J.shard = shard; J.Dl = Dl; J.H0 = H0; J.kss = kss; J.h16 = h16; J.sign = (int)radial_sign;
        for (int d = 0; d < 3; d++) { J.ns[d] = ns[d]; J.nss[d] = nss[d]; J.bn[d] = bn[d]; J.bo[d] = bo[d]; }
        J.ax = &ax; J.scale = scale; J.t0 = t0; J.pcut = pcut; J.pcond = pap_cond;
        split_run(sctx, psplit_job, &J);
        for (int k = 0; k < 2; k++) { nn_init(gpus[k]); const char *e = nn_check(); if (e) { fprintf(stderr, "cuda (GPU %d): %s\n", gpus[k], e); J.fail = 3; } }
        nn_init(gpu);
        if (J.fail) { fprintf(stderr, "\npredict: the split run failed (%s)\n", J.fail == 1 ? z3w_error() : J.fail == 2 ? z3_error() : "device error"); return 1; }
        ntiles = J.ndone; nskip = J.nskip;
    } else
    for (long i = 0; i < rd.ntiles; i++) {
        pt = now();
        pthread_mutex_lock(&rd.mu);
        while (atomic_load(&rd.filled) <= i && !atomic_load(&rd.failed)) pthread_cond_wait(&rd.cv, &rd.mu);
        pthread_mutex_unlock(&rd.mu);
        p_wait += now() - pt;
        if (atomic_load(&rd.failed)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const tile_t *t = &rd.tiles[i];
        if (ffrow) {   /* a new row of (super-)shards: the previous row's field shards are final */
            const int64_t nX = sup ? nss[2] : ns[2], nY = sup ? nss[1] : ns[1], row = t->shard / nX;
            if (ffprow >= 0 && row != ffprow) {
                const int64_t rz = ffprow / nY, ry = ffprow % nY, cz0 = rz * ffrk, slab = cz0 / FZ;
                ffw_push(&ffw, (ffw_job){.kind = 2, .slab = slab, .q0 = (int)((cz0 - slab * FZ) / FFS), .q1 = (int)((cz0 - slab * FZ + ffrk) / FFS),
                                         .y0 = (int)(ry * ffrk / FFS), .y1 = (int)((ry + 1) * ffrk / FFS)});
            }
            ffprow = row;
        }
        if (!sup && t->shard != cur_shard) {   /* new shard: flush the previous one */
            if (cur_shard >= 0) {
                if (sheet_task && sheet_write_shard(out,sheet_shard,shard,sz,sy,sx)) return 1;
                pt = now();
                if (any) { if (gpath && !WR.skip) writer_d2h(&WR, sb, dsh); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb = writer_next(&WR, sb); sbuf = WR.buf[sb]; }
                else if (WR.donefd >= 0) dprintf(WR.donefd, "%lld %lld %lld\n", (long long)(sz + soff[0]), (long long)(sy + soff[1]), (long long)(sx + soff[2]));   /* all air: done */
                p_wr += now() - pt;
                fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   ", (long long)(cur_shard + 1), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
                if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
            }
            cur_shard = t->shard; sz = cur_shard / (ns[1] * ns[2]); sy = (cur_shard / ns[2]) % ns[1]; sx = cur_shard % ns[2];
            so[0] = sz * shard; so[1] = sy * shard; so[2] = sx * shard;
            for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
            if (gpath) { if (!WR.skip) nn_zero(dsh, WR.bytes); } else memset(sbuf, 0, (size_t)shard * shard * shard);
            any = 0;
            if (sheet_task) for (size_t j=0;j<(size_t)shard*shard*shard;j++) sheet_shard[j]=NAN;
        }
        int k = (int)(i % NSLOT);
        const uint8_t *ctu = rd.buf[k];
        const int64_t tz = t->tz, ty = t->ty, tx = t->tx, o[3] = {bo[0] + tz, bo[1] + ty, bo[2] + tx};
        if (degrade) {   /* the scanner ops on this window's uint8 CT (the reader's pinned slot), its axis centres per z */
            static uint8_t *dsc; static float *dcy; if (!dsc) { dsc = malloc(6 * w3); dcy = malloc(2 * (size_t)W * sizeof *dcy); if (!dsc || !dcy) { fprintf(stderr, "predict: out of memory\n"); return 1; } }
            for (int z = 0; z < W; z++) { double cy, cx; axis_at(&ax, (double)(o[0] + z) * scale, &cy, &cx); dcy[z] = (float)(cy / scale); dcy[W + z] = (float)(cx / scale); }
            const z3_meta *cm = z3_meta_of(ct); const int chk[3] = {(int)cm->chunk[0], (int)cm->chunk[1], (int)cm->chunk[2]};
            uint64_t sd = dg_seed * 0x9e3779b97f4a7c15ull ^ (uint64_t)o[0] * 0xc2b2ae3d27d4eb4full ^ (uint64_t)o[1] * 0x165667b19e3779f9ull ^ (uint64_t)o[2] * 0x27d4eb2f165667c5ull;
            memset(dsc, 0, w3);   /* (the ignore mask: unused here) */
            scan_aug_apply(&dg, sd, (uint8_t *)rd.buf[k], nullptr, 0, dsc, W, o, dcy, dcy + W, ax.n > 0, chk, 0, dsc + w3);
        }
        {
            size_t nz = rd.nz[k]; double sum = rd.sum[k], sq = rd.sq[k];
            if (gpath) {   /* upload (pinned), statistics on the device, then the slot is free for the reader */
                pt = now();
                if (pre_i == i) {   /* prefetched: the copy stream's upload, then one device copy (the prefetch buffer free again after it) */
                    nn_stream_wait(0, ev_pre); nn_d2d_async(ctd, ctpre, w3); nn_event_record(ev_preu, 0);
                } else nn_h2d(ctd, ctu, w3);
                nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
                pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu);
                p_h2d += now() - pt;
            }
            if (nz == 0) { nskip++; pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu); continue; }
            double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
            if (pcut >= 0 && pap_cond == 1) unet_set_cond(u, (float)((pcut - mean) / sd));   /* papyrus: the cutoff in this window's input units */
            if (pcut >= 0 && pap_cond == 2) mean = pcut;                                         /* (or the input's zero) */
            /* G * ((CT - mean) / sd) + B in both explicit-input and fused-stem paths (predict --input-gain / --input-bias): the original CT
               bytes stay for masking, only the CT input channel changes */
            if (input_affine) { mean -= input_bias * sd / input_gain; sd /= input_gain; }
            for (int z = 0; z < Wz; z++) {   /* window origin relative to the axis at this slice (double on the host, small in float) */
                double cy, cx;
                axis_at(&ax, (double)(o[0] + z) * scale, &cy, &cx);
                cy /= scale; cx /= scale;
                dyo[z] = (float)((double)o[1] - cy); dxo[z] = (float)((double)o[2] - cx);
            }
            pt = now();
            if (!gpath) nn_h2d(ctd, ctu, w3);
            nn_h2d(dyd, dyo, 2 * (size_t)W * sizeof(float));
            if (pprof) { nn_sync(); p_h2d += now() - pt; pt = now(); }
            if ((xd_shared && unet_input_scratch(u, 4 * w3 * 2) != xd) || (ct_shared && unet_input_scratch2(u, w3) != ctd)) { fprintf(stderr, "predict: the network was rebuilt under its input scratch\n"); return 1; }
            if (!ct_stem) nn_pred_input_d(ctd, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, xd, h16);
            if (cfg.cin == 7 && !half) nn_zero((char *)xd + 4 * w3 * (h16 ? 2 : 4), 3 * w3 * (h16 ? 2 : 4));   /* no condition: inputs 4-6 zero */
            if (auxz) {   /* the fifth channel: the aux volume over this window (level coordinates as the CT; outside 0) */
                const z3_meta *am = z3_meta_of(auxz); int64_t ao[3], an[3]; int inb = 1;
                memset(auxh, 0, w3);
                for (int d = 0; d < 3; d++) { ao[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + W < am->shape[d] ? o[d] + W : am->shape[d]; an[d] = e - ao[d]; if (an[d] <= 0) inb = 0; }
                if (inb) {
                    uint8_t *tmp = malloc((size_t)an[0] * an[1] * an[2]);
                    if (!tmp || z3_read(auxz, ao, an, tmp, 8)) { fprintf(stderr, "predict: --aux read failed: %s\n", z3_error()); return 1; }
                    for (int64_t a = 0; a < an[0]; a++) for (int64_t b2 = 0; b2 < an[1]; b2++)
                        memcpy(auxh + ((size_t)(a + ao[0] - o[0]) * W + (b2 + ao[1] - o[1])) * W + (ao[2] - o[2]), tmp + ((size_t)a * an[1] + b2) * an[2], (size_t)an[2]);
                    free(tmp);
                }
                nn_h2d(auxd, auxh, w3);
                nn_aux_input(auxd, w3, (char *)xd + 4 * w3 * (h16 ? 2 : 4), h16);
            }
            if (ct_shared) nn_pred_ctmask(ctd, w3, ctm);   /* the forward overwrites ctd */
            static uint8_t *ctsave; static float *dyt;   /* --tta: the CT window kept for the augmented passes */
            if (tta > 1 && !ctsave) { ctsave = nn_malloc(w3); dyt = malloc(2 * (size_t)W * sizeof(float)); if (!ctsave || !dyt) { fprintf(stderr, "predict: no memory for --tta\n"); return 1; } }
            if (tta > 1) nn_d2d(ctsave, ctd, w3);
            const tta_ctx TC = {.u = u, .ctd = ctd, .ctsave = ctsave, .mean = (float)mean, .isd = (float)(1.0 / sd), .dyo = dyo, .dxo = dxo, .dyt = dyt, .dyd = dyd,
                                .axis = ax.n > 0 ? (int)radial_sign : 0, .xs = xs, .h16 = h16, .D = Wz, .W = W, .nch = cfg.cout, .ch_normal = ff_nrm};
            if (ct_shared && ctmi != ctm) nn_pred_ctmask_t(ctd, w3, ctmi, iso);
            if (ffct && ct_shared) nn_d2d_async(ctcp, ctd, w3);   /* --fit-ct reads the CT after the forward */
            if (sheet_task) {
                for (int z=0;z<W;z++) { double a[4]; sheet_parameters(sheet,o[0]+z,a); sheet_rows[4*z]=(float)(o[1]-a[0]); sheet_rows[4*z+1]=(float)(o[2]-a[1]); sheet_rows[4*z+2]=(float)(1/a[2]); sheet_rows[4*z+3]=(float)a[3]; }
                nn_sheet_input(xd,W,sheet_rows,(float)sheet->center,(float)sheet->scale,h16);
            }
            pred_axis(u, ax.n > 0, Wz, dyo, dxo);
            pred_mem_in(u, o, W);
            if (pred_view_set(u, o, Wz, W)) return 1;
            if (g_pv && tta > 1) nn_d2d(g_pvsave, g_pvd, (size_t)g_pvP * g_pvP * g_pvP);
            const float *lg;
            if (half) {   /* the network on blocks of the window: its input from the 4-channel window, its logits back to full resolution */
                nn_lowres_input(xd, xs, hdiv, hmode, lrt, x2, h16);
                const float *l2 = unet_forward_x(u, x2, xn, 0, h16);
                nn_lowres_output(l2, unet_logits_h16(u), (shape5){1, 1, W / hdiv, W / hdiv, W / hdiv}, hdiv, hmode, (float *)lrt, lgf);
                lg = lgf;
            } else lg = ct_stem ? unet_forward_ct(u, ctd, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, xs, h16)
                                : unet_forward_x(u, xd, xs, 0, h16);
            pred_mem_out(u, o);
            if (ctpre && i + 1 < rd.ntiles && atomic_load(&rd.filled) > i + 1) {   /* the next window's CT during this forward (its slot stays held
                                                                                         until that window's statistics, as with the plain upload) */
                if (pre_i >= 0) nn_stream_wait(1, ev_preu);   /* (the previous prefetch's device copy has read ctpre) */
                nn_h2d_copy_stream(ctpre, rd.buf[(i + 1) % NSLOT], w3); nn_event_record(ev_pre, 1); pre_i = i + 1;
            }
            nn_set_logits_h16(!half && unet_logits_h16(u));   /* the readers of lg below */
            const uint8_t *ctk = ct_shared && ctcp ? ctcp : ctd; (void)ctk;   /* --fit-ct: the CT window as it was before the forward */
            { static int memi = -1; if (memi < 0) memi = ufsm_env_on("UFSM_PRED_MEMINFO"); if (memi == 1) { nn_sync(); fprintf(stderr, "\npredict: GPU free after the first window %.2f GB\n", nn_mem_free() / 1e9); memi = 2; } }
            if (gn_save) unet_gn_accumulate(u);
            if (ffd && !ffzm) nn_fit_accum_dm(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
            if (ffd && !ffzm && ffw4d) nn_fit_accum_w4(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ff_w4, ffw4d, ffacc4);
            if (ffct && !ffzm) {
                nn_fit_accum_ct(ctk, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ffacc + 12 * ffn);
                nn_fit_accum_cs(lg, ctk, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ff_nrm, ff_den, ffds, ffdc, (float)(um * (1 << level)), ffacc + 13 * ffn);
            }
            if (ffd && !ffzm) for (int p = 1; p < tta; p++) {   /* --tta: the other passes into the same accumulators */
                lg = tta_pass(&TC, tta_order[p]);
                nn_fit_accum_dm(lg, ctd, ctm, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
                if (ffw4d) nn_fit_accum_w4(lg, ctd, ctm, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, ff_w4, ffw4d, ffacc4);
            }
            if (ffzm) {   /* the window interior's whole cells, finalized into the field slab (flushed when the slab is left) */
                int c0[3], c1[3], nc3[3], ok = 1; const int64_t tw[3] = {tz, ty, tx};
                for (int d = 0; d < 3; d++) {
                    int64_t lo = tw[d] + hal[d], hi = tw[d] + wn[d] - hal[d]; if (lo < 0) lo = 0; if (hi > bn[d]) hi = bn[d];
                    c0[d] = (int)((lo + ffh - 1) / ffh); c1[d] = (int)(hi / ffh); if (hi == bn[d]) c1[d] = ffF[d];   /* the box's last, possibly partial, cell */
                    nc3[d] = c1[d] - c0[d]; if (nc3[d] <= 0 || nc3[d] > (d ? ffwc : strd[0] / ffh + 1)) ok = 0;
                }
                if (ok) {
                    const int64_t slab = c0[0] / FZ;
                    while (ffslab < slab) { ffw_push(&ffw, (ffw_job){.kind = 1, .slab = ffslab}); ffslab++; }   /* after the slab's chunk copies (FIFO) */
                    if (atomic_load(&ffw.fail)) { fprintf(stderr, "predict: %s\n", z3w_error()); return 1; }
                    for (int za = 0; za < nc3[0]; za += ffzc) {   /* z chunks of the window's cells (the accumulator holds ffzc planes) */
                        const int nz = nc3[0] - za < ffzc ? nc3[0] - za : ffzc, ncz[3] = {nz, nc3[1], nc3[2]};
                        const int64_t cz = (int64_t)(c0[0] + za);
                        const size_t nl = (size_t)nz * nc3[1] * nc3[2];
                        nn_zero(ffacc, ffna * nl * 4); if (ffw4d) nn_zero(ffacc4, 9 * nl * 4);
                        const int lbox[3] = {(int)(bn[0] - cz * ffh < (int64_t)nz * ffh ? bn[0] - cz * ffh : (int64_t)nz * ffh),
                                             (int)(bn[1] - (int64_t)c0[1] * ffh < (int64_t)nc3[1] * ffh ? bn[1] - (int64_t)c0[1] * ffh : (int64_t)nc3[1] * ffh),
                                             (int)(bn[2] - (int64_t)c0[2] * ffh < (int64_t)nc3[2] * ffh ? bn[2] - (int64_t)c0[2] * ffh : (int64_t)nc3[2] * ffh)};
                        nn_fit_accum_dm(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
                        if (ffw4d) nn_fit_accum_w4(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_w4, ffw4d, ffacc4);
                        if (ffct) {
                            nn_fit_accum_ct(ctk, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ffacc + 12 * nl);
                            nn_fit_accum_cs(lg, ctk, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_nrm, ff_den, ffds, ffdc,
                                            (float)(um * (1 << level)), ffacc + 13 * nl);
                        }
                        for (int p = 1; p < tta; p++) {   /* --tta: the other passes (one chunk: the whole interior) */
                            lg = tta_pass(&TC, tta_order[p]);
                            nn_fit_accum_dm(lg, ctd, ctm, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
                            if (ffw4d) nn_fit_accum_w4(lg, ctd, ctm, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_w4, ffw4d, ffacc4);
                        }
                        const double tfa = pprof ? now() : 0;
                        const int fb = ffw_acquire(&ffw);
                        if (pprof) p_ffw += now() - tfa;
                        uint8_t *fd = ff8d + (size_t)fb * FFZ_MAXP * ffwn;
                        nn_fit_finalize(ffacc, nl, ffh, fd);   /* pc .. w, then r (unwrap) or dz dy dx dq (unwrap2) */
                        if (!ffu2) nn_fit_finalize_recto(ffacc, nl, fd + 10 * nl); else if (ffw4d) nn_fit_finalize_w4(ffacc4, nl, fd + 10 * nl);
                        if (ffct) nn_fit_finalize_ct(ffacc, nl, ffh, fd + (size_t)ffz_nb * nl, fd + (size_t)(ffz_nb + 1) * nl);   /* then ct, cs */
                        nn_offload_copy(ffw.h[fb], fd, (size_t)ffz_np * nl, 1, ffw.evs[fb], ffw.evd[fb]);   /* scattered into the slab by the fit writer */
                        ffw_push(&ffw, (ffw_job){.kind = 0, .b = fb, .c1 = c0[1], .c2 = c0[2], .nz = nz, .ny = nc3[1], .nx = nc3[2], .cz = cz, .slab = slab, .nl = nl});
                    }
                }
            }
            if (sheet_task) lg += (size_t)out_ch * w3;   /* channel c of the [c][W^3] logits (B = 1); otherwise the only one */
            if (sheet_task) {
                nn_sheet_gate((float *)lg+w3,lg,ctd,W,sheet_rows);
                nn_d2h(sheet_window,lg+w3,w3*4);
                for (int z=halo;z<W-halo;z++) { int64_t gz=tz+z-so[0]; if (gz<0 || gz>=se[0]-so[0]) continue;
                    for (int y=halo;y<W-halo;y++) { int64_t gy=ty+y-so[1]; if (gy<0 || gy>=se[1]-so[1]) continue;
                        for (int x=halo;x<W-halo;x++) { int64_t gx=tx+x-so[2]; if (gx<0 || gx>=se[2]-so[2]) continue;
                            size_t k=((size_t)z*W+y)*W+x;
                            sheet_shard[((size_t)gz*shard+gy)*shard+gx]=sheet_window[k];
                        }
                    }
                }
            }
            if (gpath && sup && fea) {   /* --feather: E (interior + B upper halo voxels) to the host, blended with the earlier spills */
                const int64_t SZ = t->shard / (nss[1] * nss[2]), SY = (t->shard / nss[2]) % nss[1], SX = t->shard % nss[2];
                const size_t e3 = (size_t)FEN * FEN * FEN;
                uint8_t *fdst = fext;
                if (ct_shared && w3 >= e3 && ((const uint8_t *)lg + w3 * 4 <= ctd || (const uint8_t *)lg >= ctd + e3)) fdst = ctd;
                else if (!fdst && !(fdst = fext = nn_malloc_try(e3))) { fprintf(stderr, "predict: --feather: no device memory for E (%.2f GB)\n", e3 / 1e9); return 1; }
                if (ct_shared) nn_pred_place_m(lg, ctmi, W, halo - fea, -halo, -halo, -halo, FEN, FEN, FEN, FEN, fdst);
                else nn_pred_place(lg, ctd, W, halo - fea, -halo, -halo, -halo, FEN, FEN, FEN, FEN, fdst);
                const int fk = (int)(fe_n++ & 1);
                pthread_mutex_lock(&FW.mu);
                while (FW.busy[fk] && !FW.failed) pthread_cond_wait(&FW.cv, &FW.mu);   /* (its previous window blended) */
                const int ff = FW.failed; pthread_mutex_unlock(&FW.mu);
                if (ff) { fprintf(stderr, "predict: --feather: %s\n", ff == 2 ? "out of host memory for the spills" : z3w_error()); return 1; }
                if (pprof) { p_copy += now() - pt; pt = now(); }
                nn_d2h(FW.fh[fk], fdst, e3);
                const char *e = nn_check();
                if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
                if (pprof) { p_gpu += now() - pt; pt = now(); }
                pthread_mutex_lock(&FW.mu);
                FW.busy[fk] = 1; FW.q[(FW.qh + FW.qn) % 2] = (fe_job){fk, SZ, SY, SX, t->noout}; FW.qn++;
                pthread_cond_broadcast(&FW.cv); pthread_mutex_unlock(&FW.mu);
                ntiles++;
                fprintf(stderr, "\rwindow %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, rd.ntiles, ntiles, nskip, now() - t0);
                continue;
            }
            if (gpath && sup) {   /* every shard of this super-shard from the one window, written as it is placed */
                const int64_t SZ = t->shard / (nss[1] * nss[2]), SY = (t->shard / nss[2]) % nss[1], SX = t->shard % nss[2];
                for (int64_t qz = SZ * kss; qz < SZ * kss + kss && qz < ns[0]; qz++) for (int64_t qy = SY * kss; qy < SY * kss + kss && qy < ns[1]; qy++)
                for (int64_t qx = SX * kss; qx < SX * kss + kss && qx < ns[2]; qx++) {
                    const int64_t qo[3] = {qz * shard, qy * shard, qx * shard}, qe[3] = {qo[0] + shard < bn[0] ? qo[0] + shard : bn[0], qo[1] + shard < bn[1] ? qo[1] + shard : bn[1], qo[2] + shard < bn[2] ? qo[2] + shard : bn[2]};
                    if (WR.skip) { if (writer_submit(&WR, sb, qz, qy, qx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb = writer_next(&WR, sb); sbuf = WR.buf[sb]; continue; }   /* --no-out */
                    nn_zero(dsh, WR.bytes);
                    if (ct_shared) nn_pred_place_m(lg, ctmi, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else nn_pred_place(lg, ctd, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    writer_d2h(&WR, sb, dsh);
                    if (writer_submit(&WR, sb, qz, qy, qx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
                    sb = writer_next(&WR, sb); sbuf = WR.buf[sb];
                }
                const char *e = nn_check();
                if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
                if (pprof) { nn_sync(); p_gpu += now() - pt; }
                ntiles++;
                fprintf(stderr, "\rsuper-shard %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, rd.ntiles, ntiles, nskip, now() - t0);
                continue;
            }
            if (gpath) {   /* interior straight into the device shard buffer (--no-out: nothing placed) */
                if (WR.skip) ;
                else if (ct_shared) nn_pred_place_m_d(lg, ctmi, Wz, W, hz, halo, (int)(tz - so[0]), (int)(ty - so[1]), (int)(tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]), shard, dsh);
                else nn_pred_place_d(lg, ctd, Wz, W, hz, halo, (int)(tz - so[0]), (int)(ty - so[1]), (int)(tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]), shard, dsh);
                const char *e = nn_check();
                if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
                if (pprof) { nn_sync(); p_gpu += now() - pt; }
                any = 1; ntiles++;
                continue;
            }
            nn_pred_output(lg, ctd, w3, pud);      /* channel 0 = recto */
            nn_d2h(pu, pud, w3);
            if (pprof) { p_gpu += now() - pt; pt = now(); }
            pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu);   /* slot free: the window is on the device */
            const char *e = nn_check();
            if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
            /* interior into the shard buffer */
            for (int z = halo; z < W - halo; z++) {
                int64_t gz = tz + z - so[0]; if (gz < 0 || gz >= se[0] - so[0]) continue;
                for (int y = halo; y < W - halo; y++) {
                    int64_t gy = ty + y - so[1]; if (gy < 0 || gy >= se[1] - so[1]) continue;
                    for (int x = halo; x < W - halo; x++) {
                        int64_t gx = tx + x - so[2]; if (gx < 0 || gx >= se[2] - so[2]) continue;
                        size_t k = ((size_t)z * W + y) * W + x;
                        sbuf[((size_t)gz * shard + gy) * shard + gx] = pu[k];
                    }
                }
            }
            any = 1; ntiles++;
            if (pprof) p_copy += now() - pt;
        }
    }
    if (fea) {   /* the feather thread's last windows */
        const double tw = now(); const int ff = fe_wait_idle(&FW);
        pthread_mutex_lock(&FW.mu); FW.stop = 1; pthread_cond_broadcast(&FW.cv); pthread_mutex_unlock(&FW.mu); pthread_join(FW.th, nullptr);
        if (ff) { fprintf(stderr, "predict: --feather: %s\n", ff == 2 ? "out of host memory for the spills" : z3w_error()); return 1; }
        if (pprof) p_copy += now() - tw;
    }
    if (pprof) fprintf(stderr, "\npredict profile (s): main: wait-reader %.2f h2d %.2f gpu(fwd+d2h) %.2f interior-copy %.2f writer-wait %.2f fit-writer-wait %.2f | reader: read %.2f stats %.2f slot-wait %.2f | wall %.2f, %ld tiles\n",
                       p_wait, p_h2d, p_gpu, p_copy, p_wr, p_ffw, rd.t_read, rd.t_stats, rd.t_wait, now() - t0, ntiles);
    if (!sup && cur_shard >= 0 && any) { if (gpath && !WR.skip) writer_d2h(&WR, sb, dsh); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } }
    else if (!sup && cur_shard >= 0 && WR.donefd >= 0) dprintf(WR.donefd, "%lld %lld %lld\n", (long long)(sz + soff[0]), (long long)(sy + soff[1]), (long long)(sx + soff[2]));
    if (sheet_task && cur_shard>=0 && sheet_write_shard(out,sheet_shard,shard,sz,sy,sx)) return 1;
    if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
    { const double tf = now();
      if (writer_finish(&WR, wth)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
      if (pprof) fprintf(stderr, "predict profile (s): writer: %.2f compressing %ld shards (%.0f MB/s), tail after the last window %.2f\n",
                         WR.t_write, WR.n_write, WR.t_write > 0 ? WR.n_write * (double)WR.bytes / WR.t_write / 1e6 : 0., now() - tf); }
    if (ffzm) {   /* the last field slabs, the axis table and fields.json (format zarr) */
        for (; ffslab * FZ < ffF[0]; ffslab++) ffw_push(&ffw, (ffw_job){.kind = 1, .slab = ffslab});
        const double tfin = now();
        if (ffw_on && ffw_finish(&ffw, ffw_th)) { fprintf(stderr, "predict: %s\n", z3w_error()); return 1; }
        if (pprof) fprintf(stderr, "predict profile (s): fit writer tail (the last field slabs) %.2f\n", now() - tfin);
        for (int k = 0; k < ffz_np; k++) { z3w_close(ffzw[k]); free(ffsl[k]); }
        free(ffw.done); free(ffw.rg.own);
        nn_free(ffacc); nn_free(ffacc4); nn_free(ff8d); nn_host_free(ff8h);
        char p[1400]; snprintf(p, sizeof p, "%s/fields.json", ffz); FILE *f = fopen(p, "w");
        if (!f) { fprintf(stderr, "predict: cannot write %s\n", p); return 1; }
        fprintf(f, "{\"format\": \"zarr\", \"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.9g, ", (long long)bo[0], (long long)bo[1], (long long)bo[2], ffF[0], ffF[1], ffF[2], ffh, um * (1 << level));
        if (ffu2) { if (ffw4d) fprintf(f, "\"w4_d\": %d, ", ffw4d); } else fprintf(f, "\"recto\": true, ");
        fprintf(f, "\"axis\": [");
        for (int64_t z = 0; z <= bn[0]; z++) { double cy = 0, cx = 0; if (ax.n) axis_at(&ax, (double)(bo[0] + z) * (1 << level), &cy, &cx); fprintf(f, "%s[%.3f, %.3f]", z ? ", " : "", cy / (1 << level) - bo[1], cx / (1 << level) - bo[2]); }
        fprintf(f, "]}\n"); fclose(f);
        fprintf(stderr, "predict: fit fields %dx%dx%d (h %d) written to %s (zarr)\n", ffF[0], ffF[1], ffF[2], ffh, ffz);
    } else if (ffd) {   /* the fit fields: means over material (CT > 0) of the pooled heads, material fraction, max recto */
        float *acc = malloc(ffna * ffn * 4); nn_sync(); nn_d2h(acc, ffacc, ffna * ffn * 4); nn_free(ffacc);
        uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(ffn, 1);
        fit_finalize(acc, ffn, ffh, a);
        const int na = (int)bn[0] + 1; float *cyb = malloc(na * sizeof(float)), *cxb = malloc(na * sizeof(float));
        for (int z = 0; z < na; z++) { double cy = 0, cx = 0; if (ax.n) axis_at(&ax, (double)(bo[0] + z) * (1 << level), &cy, &cx); cyb[z] = (float)(cy / (1 << level) - bo[1]); cxb[z] = (float)(cx / (1 << level) - bo[2]); }
        const long long org[3] = {bo[0], bo[1], bo[2]};
        /* Preserve the recto head for local fitting as well as the normal/density/phase heads.
           acc[11] is already the maximum recto probability over each field cell. */
        uint8_t *recto = ffu2 ? nullptr : malloc(ffn);
        if (!ffu2 && !recto) return 1;
        if (recto) for (size_t i = 0; i < ffn; i++) recto[i] = (uint8_t)lroundf(255.f * fminf(1.f, fmaxf(0.f, acc[11 * ffn + i])));
        uint8_t *a14[14] = {0}; for (int k = 0; k < 10; k++) a14[k] = a[k];
        if (ffw4d) {   /* task unwrap2: the relative winding mod 4 planes, finalized on the device */
            uint8_t *d4 = nn_malloc(4 * ffn); for (int k = 0; k < 4; k++) a14[10 + k] = malloc(ffn);
            nn_fit_finalize_w4(ffacc4, ffn, d4); nn_sync(); for (int k = 0; k < 4; k++) nn_d2h(a14[10 + k], d4 + k * ffn, ffn);
            nn_free(d4); nn_free(ffacc4);
        }
        const int saved = fit_fields_save_ex(ffd, org, ffF, ffh, um * (1 << level), a14, ffw4d, recto, cyb, cxb, na); free(recto);
        for (int k = 10; k < 14; k++) free(a14[k]);
        if (saved) { fprintf(stderr, "predict: cannot write the fit fields to %s\n", ffd); return 1; }
        if (ffct) {   /* ct.raw / cs.raw (box-shaped, as ufsm ct-fields writes them) */
            uint8_t *pl = malloc(ffn); int bad = !pl;
            for (int q = 0; q < 2 && !bad; q++) {
                for (size_t i = 0; i < ffn; i++) {
                    const float cnt = acc[i], v = q == 0 ? (cnt > 0 ? acc[12 * ffn + i] / cnt : 0.f) : (acc[13 * ffn + i] > 0 ? acc[13 * ffn + i] * 8.f / ffh : 0.f);
                    pl[i] = v <= 0 ? 0 : q == 1 && v < 1 ? 1 : v > 255 ? 255 : (uint8_t)lroundf(v);
                }
                char p[1400]; snprintf(p, sizeof p, "%s/%s.raw", ffd, q ? "cs" : "ct"); FILE *f = fopen(p, "wb");
                bad = !f || fwrite(pl, 1, ffn, f) != ffn; if (f && fclose(f)) bad = 1;
            }
            free(pl); if (bad) { fprintf(stderr, "predict: cannot write the CT planes to %s\n", ffd); return 1; }
        }
        fprintf(stderr, "predict: fit fields %dx%dx%d (h %d) written to %s\n", ffF[0], ffF[1], ffF[2], ffh, ffd);
        for (int k = 0; k < 10; k++) free(a[k]);
        free(acc); free(cyb); free(cxb);
    }
    if (gpath) { for (int k = 0; k < WR.nb; k++) nn_host_free(WR.buf[k]); nn_free(dsh); } else for (int k = 0; k < WR.nb; k++) free(WR.buf[k]);
    nn_free(stsc); nn_free(x2); nn_free(lgf); nn_free(lrt);
    pthread_join(rth, nullptr);
    for (int k = 0; k < NSLOT; k++) { if (gpath) nn_host_free(rd.buf[k]); else free(rd.buf[k]); }
    free(rd.tiles);
    fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   \n", (long long)(ns[0] * ns[1] * ns[2]), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
    z3w_close(w);
    if (WR.donefd >= 0) close(WR.donefd);
    free(sskip); free(fneed); feather_free(&FE); if (fhost) nn_host_free(fhost); if (fext) nn_free(fext);
    if (gn_save) {
        const size_t ns = unet_gn_nstats(u); float *h = malloc(ns * 4); const long nw = unet_gn_pooled(u, h);
        FILE *f = fopen(gn_save, "wb");
        if (!nw || !f) { fprintf(stderr, "predict: no GroupNorm statistics to save (%ld windows)\n", nw); return 1; }
        fprintf(f, "UFSMGN{\"nstats\":%zu,\"windows\":%ld,\"window\":%d,\"ema\":%d,\"ckpt\":\"%s\"}\n", ns, nw, W, use_ema, ckpt);
        if (fwrite(h, 4, ns, f) != ns || fclose(f)) { fprintf(stderr, "predict: cannot write %s\n", gn_save); return 1; }
        fprintf(stderr, "GroupNorm statistics of %ld windows -> %s\n", nw, gn_save); free(h);
    }
    if (split) {
        for (int k = 0; k < 2; k++) { pside_t *S = &J.s[k]; nn_init(gpus[k]); nn_free(S->ctm); nn_free(S->dsh); nn_free(S->stsc); nn_free(S->red); nn_free(S->dyd); free(S->rows); free(S->own); }
        nn_init(gpus[1]); unet_free(u1); nn_init(gpu); split_free(sctx);
    }
    if (nwork > 1) { fprintf(stderr, "gpu %d: %ld tiles in %.0fs\n", gpu, ntiles, now() - t0); z3_close(ct); store_close(s); unet_free(u); return 0; }   /* worker: the parent builds the pyramid */
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, ash, shard, q, 0, nthreads, attrs)) return 1;
    if (pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs)) return 1;
    fprintf(stderr, "wrote %s (%ld tiles in %.0fs)\n", out, ntiles, now() - t0);
    z3_close(ct); store_close(s); unet_free(u);
    free(sheet_window); free(sheet_shard); free(sheet_rows); sheet_free(sheet);
    return 0;
}
