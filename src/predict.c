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
#include "vcgpu.h"
#include "gpudec.h"
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
#include "latvol.h"
#include "latvol_read.h"
#include "embz.h"
#include "latfit.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/syscall.h>
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
    int lv; lv_chunk_ref *lvref[NSLOT]; int lvn[NSLOT], lvmax; size_t lvbytes[NSLOT];   /* a latvol CT array read as cells: each slot holds the window's packed chunk streams (lv_read_box), no decode on the host */
    zr *lz; const long long *lzo; int lzc, lzt;   /* (lzt handles of the store, one decoding thread each) */   /* --latents: the window's stored cells instead of its CT (nz[k]: 0 when every cell is air) */
    uint8_t *dbuf[NSLOT]; void **ctw; int nctw, ngpuw; int64_t csh[3];   /* UFSM_PRED_DEVREAD: device windows, filled by nctw chunk workers (nn_ctw_*); csh the array shape */
    _Atomic long cnext; long nch; int64_t c0[3], cn[3], tz_o[3]; int cslot; atomic_long ndev, nhost, nskip;   /* (the window being read: its chunk grid) */
} reader_t;
/* one device-read worker: chunks of the current window decoded (the GPU's substreams on the device) and gathered into its device slot */
typedef struct { reader_t *r; int w; } ctw_arg;
static void *ctw_work(void *a) {
    ctw_arg *A = a; reader_t *r = A->r; void *w = r->ctw[A->w]; uint8_t *hc = nn_ctw_host(w), *dc = A->w < r->ngpuw ? nn_ctw_dev(w) : nullptr;
    z3_thread_cpu_decode(!dc);   /* workers ngpuw.. decode on the CPU (their chunks uploaded whole): the CPU's rate adds to the GPU's */
    const int64_t n128[3] = {128, 128, 128}, wn[3] = {r->Wz, r->W, r->W}; const int k = r->cslot;
    for (long j; (j = atomic_fetch_add(&r->cnext, 1)) < r->nch && !atomic_load(&r->failed);) {
        const int64_t co[3] = {(r->c0[0] + j / (r->cn[1] * r->cn[2])) * 128, (r->c0[1] + j / r->cn[2] % r->cn[1]) * 128, (r->c0[2] + j % r->cn[2]) * 128};
        uint32_t dm = 0;
        if (z3_read_sub_dev(r->ct, co, n128, hc, 1, 0, dc, &dm)) { atomic_store(&r->failed, 1); break; }
        int host = dm != 0xffffffffu;
        if (host && !dm) {   /* everything on the host: nothing to do when it is all fill (0; the window is zeroed) */
            const uint64_t *q = (const uint64_t *)hc; uint64_t any = 0; for (size_t i = 0; i < ((size_t)1 << 21) / 8 && !any; i++) any |= q[i];
            if (!any) { atomic_fetch_add(&r->nskip, 1); continue; }
        }
        atomic_fetch_add(host ? &r->nhost : &r->ndev, 1);
        if (nn_ctw_put(w, dm, host, co, r->tz_o, wn, r->csh, r->dbuf[k])) { atomic_store(&r->failed, 1); break; }
    }
    return nullptr;
}
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
        if (r->lz) {   /* --latents: Wz/4 x W/4 x W/4 cells of C + 1 bytes (absent / outside: 0, air) */
            const int64_t oc[3] = {(o[0] - r->lzo[0]) / 4, (o[1] - r->lzo[1]) / 4, (o[2] - r->lzo[2]) / 4}, nc[3] = {r->Wz / 4, r->W / 4, r->W / 4};
            if (zr_read_i8_mt(r->lz, r->lzt, oc, nc, (int8_t *)r->buf[k])) atomic_store(&r->failed, 1);
            else {
                const size_t nb = (size_t)nc[0] * nc[1] * nc[2] * (r->lzc + 1); const uint64_t *b8 = (const uint64_t *)r->buf[k]; uint64_t any = 0;
                for (size_t j = 0; j < nb / 8 && !any; j++) any |= b8[j];
                for (size_t j = nb / 8 * 8; j < nb && !any; j++) any |= r->buf[k][j];
                r->nz[k] = any != 0; r->t_read += now() - t1;
            }
        } else if (r->lv) {
            const int nr = lv_read_box(r->ct, o, n, r->buf[k], w3, r->lvref[k], r->lvmax, &r->lvbytes[k], r->nthreads);
            if (nr < 0) { fprintf(stderr, "\npredict: latvol window read failed (%s)\n", nr == -2 ? "the streams exceed the window buffer" : z3_error()); atomic_store(&r->failed, 1); }
            else { r->lvn[k] = nr; r->nz[k] = nr != 0; r->t_read += now() - t1; }
        } else if (r->dbuf[0]) {   /* UFSM_PRED_DEVREAD: the window decoded into device slot k */
            if (nn_ctw_zero(r->ctw[0], r->dbuf[k], w3)) atomic_store(&r->failed, 1);
            int empty = 0;
            for (int d = 0; d < 3; d++) { const int64_t lo = o[d] < 0 ? 0 : o[d], hi = o[d] + n[d] > r->csh[d] ? r->csh[d] : o[d] + n[d];
                                          if (hi <= lo) empty = 1; else { r->c0[d] = lo / 128; r->cn[d] = (hi + 127) / 128 - r->c0[d]; } r->tz_o[d] = o[d]; }
            if (!empty) {
                r->nch = r->cn[0] * r->cn[1] * r->cn[2]; atomic_store(&r->cnext, 0); r->cslot = k;
                pthread_t th[256]; ctw_arg A[256]; const int nt = r->nctw < 256 ? r->nctw : 256;
                for (int t = 0; t < nt; t++) { A[t] = (ctw_arg){r, t}; pthread_create(&th[t], nullptr, ctw_work, &A[t]); }
                for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
            }
            r->t_read += now() - t1;
        } else
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
/* the CT prefetcher (an HTTP root with --cache): the inner chunks of the windows ahead of the reader fetched into the local cache by nt
   threads (z3_prefetch_chunk, no decode), so the reader's z3_read decodes from the cache instead of fetching and decoding in turn:
   a cold 19 Gvox pod run read 14 s against 7.7 s warm. UFSM_PRED_PREFETCH=N windows ahead (2; 0 off) */
typedef struct { reader_t *rd; int ahead, nt; atomic_int stop; long nfetch; double t_fetch; int64_t c0[3], c1[3]; _Atomic long next; atomic_int bad; } prefetch_t;
static void *prefetch_work(void *a) {
    prefetch_t *P = a; const int64_t ny = P->c1[1] - P->c0[1], nx = P->c1[2] - P->c0[2], n = (P->c1[0] - P->c0[0]) * ny * nx; long nf = 0;
    for (long k; (k = atomic_fetch_add(&P->next, 1)) < n && !atomic_load(&P->stop);) {
        const int r = z3_prefetch_chunk(P->rd->ct, P->c0[0] + k / (ny * nx), P->c0[1] + k / nx % ny, P->c0[2] + k % nx);
        if (r < 0) atomic_store(&P->bad, 1); else nf += r;
    }
    return (void *)(intptr_t)nf;
}
static void *prefetch_main(void *a) {
    prefetch_t *P = a; reader_t *r = P->rd; const z3_meta *m = z3_meta_of(r->ct);
    for (long i = 0; i < r->ntiles && !atomic_load(&P->stop); i++) {
        pthread_mutex_lock(&r->mu);
        while (i >= atomic_load(&r->filled) + P->ahead && !atomic_load(&P->stop) && !atomic_load(&r->failed)) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        if (atomic_load(&P->stop) || atomic_load(&r->failed)) break;
        if (i < atomic_load(&r->filled)) continue;   /* (the reader got there first) */
        const tile_t *t = &r->tiles[i]; const int64_t o[3] = {r->bo[0] + t->tz, r->bo[1] + t->ty, r->bo[2] + t->tx}, n[3] = {r->Wz, r->W, r->W};
        int empty = 0;
        for (int d = 0; d < 3; d++) { int64_t lo = o[d] < 0 ? 0 : o[d], hi = o[d] + n[d] > m->shape[d] ? m->shape[d] : o[d] + n[d]; if (hi <= lo) empty = 1;
                                      P->c0[d] = lo / m->chunk[d]; P->c1[d] = (hi + m->chunk[d] - 1) / m->chunk[d]; }
        if (empty) continue;
        const double t0 = now(); atomic_store(&P->next, 0); pthread_t th[64]; const int nt = P->nt < 64 ? P->nt : 64;
        for (int k = 0; k < nt; k++) pthread_create(&th[k], nullptr, prefetch_work, P);
        for (int k = 0; k < nt; k++) { void *rv; pthread_join(th[k], &rv); P->nfetch += (long)(intptr_t)rv; }
        P->t_fetch += now() - t0;
    }
    return nullptr;
}
/* shard writer: the main thread hands a filled shard buffer to a writer thread and continues into the next buffer of a ring of nb
   (2 by default; one window per k^3 shards: up to 2 k^3, UFSM_PRED_WBUF, so the shards of a window are compressed while the next window
   runs instead of the main thread waiting on each shard's predecessor); buffers are written in submission (ring) order */
#define WBUF_MAX 32
#define WRITERS_MAX 8   /* writer threads: shards with fewer chunks than --threads (--shard 128: one 128^3 chunk, compressed by one thread whatever
                           --threads says, 144 MB/s: 64 shards of a 640^3 window took 0.9 s against 0.15 s of GPU work) are written several at a time */
typedef struct { z3w *w; int nthreads; size_t bytes; uint8_t *buf[WBUF_MAX]; int64_t sz[WBUF_MAX], sy[WBUF_MAX], sx[WBUF_MAX]; int pending[WBUF_MAX]; int nb, next;
                 atomic_int failed; pthread_mutex_t mu; pthread_cond_t cv; int stop;
                 int gpu, async, skip; void *evs[WBUF_MAX], *evd[WBUF_MAX]; double t_write; long n_write;
                 int nw, wt, taken[WBUF_MAX]; pthread_t wth[WRITERS_MAX];   /* nw writer threads (writer_start), wt threads inside each shard's write; taken: a writer holds buffer k */
                 int64_t off[3]; int donefd; atomic_long nsub, ndone;
                 void (*post)(void *ctx, const uint8_t *buf, int64_t sz, int64_t sy, int64_t sx); void *pctx;
                 int venc, vn; size_t voff, vcap; float vq; atomic_long vfall; } writer_t;   /* venc: the GPU's levels at buf + voff (vcgpu.h, shard vn^3, capacity vcap) */   /* post: after a shard's write, with its buffer (the GPU pyramid's pieces follow the shard's bytes) */   /* off: shard index offset (--full-grid); donefd: the --resume log (-1 none) */   /* (UFSM_PRED_PROF: seconds compressing / writing, shards) */   /* async: buffer k arrives by a copy on the offload stream (evd[k]) */
/* a shard from the GPU's levels (vcgpu.h): its chunks tokenised and entropy-coded by wt threads (vcg_encode_levels: volcomp_encode's
   stream); levels above the capacity: from the shard's bytes (z3w_write_shard) */
typedef struct { const writer_t *W; const uint8_t *b; vcg_layout L; uint8_t **enc; size_t *len; int nc; _Atomic int next; atomic_int bad; } wl_job;
static void *wl_work(void *a) {
    wl_job *J = a; const uint8_t *b = J->b; uint8_t *e = nullptr;
    const int32_t *dc = (const int32_t *)(b + J->L.o_dc), *lv = (const int32_t *)(b + J->L.o_lv); const uint32_t *cnt = (const uint32_t *)(b + J->L.o_cnt), *off = (const uint32_t *)(b + J->L.o_off);
    const int16_t *cv = (const int16_t *)(b + J->L.o_cv); const uint16_t *pos = (const uint16_t *)(b + J->L.o_pos);
    for (int c; (c = atomic_fetch_add(&J->next, 1)) < J->nc && !atomic_load(&J->bad);) {
        int any = 0; for (int k = 0; k < 512 && !any; k++) any = cv[c * 512 + k] != 0;   /* (fill 0: a chunk of constant-0 blocks is omitted) */
        if (!any) continue;
        if (!e && !(e = malloc(VCG_ENCODE_BOUND))) { atomic_store(&J->bad, 1); break; }
        size_t n;
        if (vcg_encode_levels(J->W->vq, dc + c * 512, cnt + c * 512, off + c * 512, lv, pos, e, VCG_ENCODE_BOUND, &n) || !(J->enc[c] = malloc(n))) { atomic_store(&J->bad, 1); break; }
        memcpy(J->enc[c], e, n); J->len[c] = n;
    }
    free(e); return nullptr;
}
static int writer_write_levels(writer_t *W, int k, int64_t sz, int64_t sy, int64_t sx, int nt) {
    wl_job J = {.W = W, .b = W->buf[k] + W->voff}; vcg_layout_of(&J.L, W->vn, W->vcap);
    uint64_t tot; memcpy(&tot, J.b, 8);
    if (tot > W->vcap) { atomic_fetch_add(&W->vfall, 1); return z3w_write_shard(W->w, sz, sy, sx, W->buf[k], nt); }
    J.nc = z3w_shard_chunks(W->w); J.enc = calloc((size_t)J.nc, sizeof(uint8_t *)); J.len = calloc((size_t)J.nc, sizeof(size_t));
    if (!J.enc || !J.len) return -1;
    nt = nt > J.nc ? J.nc : nt; nt = nt > 64 ? 64 : nt < 1 ? 1 : nt;
    pthread_t th[64]; for (int i = 1; i < nt; i++) pthread_create(&th[i], nullptr, wl_work, &J);
    wl_work(&J); for (int i = 1; i < nt; i++) pthread_join(th[i], nullptr);
    const int rc = atomic_load(&J.bad) ? -1 : z3w_write_shard_enc(W->w, sz, sy, sx, J.enc, J.len);
    for (int i = 0; i < J.nc; i++) free(J.enc[i]);
    free(J.enc); free(J.len);
    return rc;
}
static void *writer_main(void *arg) {
    writer_t *W = arg;
    if (W->async) nn_init(W->gpu);
    for (;;) {
        pthread_mutex_lock(&W->mu);
        int k = -1;
        while (k < 0 && !W->stop) {   /* the oldest submitted buffer no writer holds (ring order from next) */
            for (int j = 0; j < W->nb && k < 0; j++) { const int q = (W->next + j) % W->nb; if (W->pending[q] && !W->taken[q]) k = q; }
            if (k < 0) pthread_cond_wait(&W->cv, &W->mu);
        }
        if (k < 0) { pthread_mutex_unlock(&W->mu); return nullptr; }
        W->taken[k] = 1; W->next = (k + 1) % W->nb;
        pthread_mutex_unlock(&W->mu);
        if (W->async) nn_event_sync(W->evd[k]);   /* the shard's device-to-host copy (queued without blocking the main thread) */
        const double tw = now();
        if (!W->skip && (W->venc ? writer_write_levels(W, k, W->sz[k] + W->off[0], W->sy[k] + W->off[1], W->sx[k] + W->off[2], W->wt > 0 ? W->wt : W->nthreads)
                                 : z3w_write_shard(W->w, W->sz[k] + W->off[0], W->sy[k] + W->off[1], W->sx[k] + W->off[2], W->buf[k], W->wt > 0 ? W->wt : W->nthreads))) atomic_store(&W->failed, 1);   /* skip: --no-out 1 */
        else if (W->donefd >= 0) dprintf(W->donefd, "%lld %lld %lld\n", (long long)(W->sz[k] + W->off[0]), (long long)(W->sy[k] + W->off[1]), (long long)(W->sx[k] + W->off[2]));   /* (O_APPEND: whole lines across GPU workers) */
        if (W->post && !atomic_load(&W->failed)) W->post(W->pctx, W->buf[k], W->sz[k] + W->off[0], W->sy[k] + W->off[1], W->sx[k] + W->off[2]);
        const double dw = now() - tw;
        pthread_mutex_lock(&W->mu); W->t_write += dw; W->n_write++; W->pending[k] = 0; W->taken[k] = 0; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
        atomic_fetch_add(&W->ndone, 1);
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
    W->sz[k] = sz; W->sy[k] = sy; W->sx[k] = sx; W->pending[k] = 1; atomic_fetch_add(&W->nsub, 1); pthread_cond_broadcast(&W->cv);
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
    fe_job q[2]; int qh, qn, stop, failed; atomic_long done;   /* done: windows blended and handed to the writer */
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
        W->busy[j.k] = 0; W->qh = (W->qh + 1) % 2; W->qn--; atomic_fetch_add(&W->done, 1);
        pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
}
static int fe_wait_idle(fe_worker *W) {   /* every queued window blended and handed to the writer; the failure code */
    pthread_mutex_lock(&W->mu);
    while (W->qn) pthread_cond_wait(&W->cv, &W->mu);
    const int f = W->failed; pthread_mutex_unlock(&W->mu);
    return f;
}
/* level 1 of the output pyramid behind the --feather windows (UFSM_PYR_PIPE, default on): once the windows have moved past level-0
   shard layer 2 Z + 1 (global shard index) and the feather thread and the writer have written everything handed over until then, a
   thread (nice 10) builds level-1 shard layer Z from the level-0 shards on disk (pyramid_build_level_z), as the build after the run
   would; the layers it has not reached are built after the run. Same bytes; the ~15% of a whole-scroll ink pass that level 1 took
   (pod P0800: 171 of 1375 s) overlaps the windows */
typedef struct { int64_t z; long fe_t; } pp_job;
typedef struct {
    const char *out; double um; const int64_t *shape0; int shard, nthreads; float q; const char *attrs;
    writer_t *WR; fe_worker *FW;
    int64_t nz1, next; uint8_t *done; pp_job *jq; int64_t qh, qn; int stop, failed; double busy;
    pthread_mutex_t mu; pthread_cond_t cv; pthread_t th;
} pyr_pipe;
static void *pyr_main(void *arg) {
    pyr_pipe *P = arg;
    if (setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10)) { }   /* (the windows' reader / blend / writer threads first; inherited by the build's threads) */
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (P->qh == P->qn && !P->stop) pthread_cond_wait(&P->cv, &P->mu);
        if (P->qh == P->qn) { pthread_mutex_unlock(&P->mu); return nullptr; }
        const pp_job j = P->jq[P->qh];
        pthread_mutex_unlock(&P->mu);
        while (atomic_load(&P->FW->done) < j.fe_t && !atomic_load(&P->WR->failed)) { pthread_mutex_lock(&P->FW->mu); const int f = P->FW->failed; pthread_mutex_unlock(&P->FW->mu); if (f) break; usleep(5000); }
        const long sub = atomic_load(&P->WR->nsub);
        while (atomic_load(&P->WR->ndone) < sub && !atomic_load(&P->WR->failed)) usleep(5000);
        const double t = now();
        const int r = pyramid_build_level_z(P->out, P->um, 1, P->shape0, P->shard, P->q, 0, P->nthreads, P->attrs, j.z, j.z + 1, 1);
        pthread_mutex_lock(&P->mu);
        P->busy += now() - t; if (r) P->failed = 1; else P->done[j.z] = 1; P->qh++;
        pthread_mutex_unlock(&P->mu);
    }
}
/* the main loop at a window of global level-0 shard layer gz: queue the level-1 layers below it */
static void pyr_advance(pyr_pipe *P, int64_t gz, long fe_t) {
    if (!P) return;
    pthread_mutex_lock(&P->mu);
    while (P->next < P->nz1 && 2 * P->next + 1 < gz) { P->jq[P->qn++] = (pp_job){P->next, fe_t}; P->next++; }
    pthread_cond_broadcast(&P->cv); pthread_mutex_unlock(&P->mu);
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

static int writer_finish(writer_t *W) {
    pthread_mutex_lock(&W->mu);
    for (int k = 0; k < W->nb; k++) while (W->pending[k]) pthread_cond_wait(&W->cv, &W->mu);   /* every buffer of the ring written */
    W->stop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    for (int i = 0; i < W->nw; i++) pthread_join(W->wth[i], nullptr);
    return atomic_load(&W->failed);
}
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* the multi-head path of latent-grid nets (predict --heads, 2026-11-11): one trunk forward per window, every requested head placed per
   shard straight from the cell logits (nn_lat_head_place: upsampling, sub-cell term, fine stem, rv fusion) into its own store <out>/<name>,
   each a pyramid as the single-output path writes it, with its own writer threads and pinned ring; the single output of a latent net
   with a fine stem or --rv 2 takes the same path (one head, the main store) */
#define PHEAD_MAX 24
/* the GPU pyramid of a head store (predict --heads, UFSM_PRED_GPU_PYR=0: off): each level-0 shard, complete on the device, is pooled 2x on the
   GPU down the levels (nn_pool2_u8: the CPU pyramid's mean) into pieces after the shard's bytes, copied to the host with it; the head's
   writer thread, after writing the level-0 shard, hands the pieces here. A piece of >= 128^3 voxels (level l with shard >> l >= 128) is
   encoded into its level-l chunks at once; smaller pieces gather raw in their 128^3 chunk. Shards (and raw chunks) of a level are final
   once every level-0 shard layer under them is: the windows run in z order, so the main thread marks the lowest level-0 shard z any
   later window can submit (pyacc_mark), and a layer is done when marked past and all its submitted shards have been added. The CPU
   pyramid decoded level l - 1 back from its store (lossy at q > 0) and pooled that; this pools the exact bytes, so at q 0 the levels are
   the same bytes, at q > 0 they are one lossy step closer to the exact pooling. */
typedef struct { uint8_t **enc; size_t *len; uint8_t **raw; } pya_shard;
typedef struct { z3w *w; int64_t shape[3], ns[3]; int pe; pya_shard **sh; int64_t next_layer, next_clayer; } pya_level;
typedef struct { pya_level L[8]; int nlev, shard, nc, nthreads; size_t poff[9]; pthread_mutex_t mu; int64_t nz0, mark, zdone; long *nsub, *nadd; atomic_int failed;
                 double t_enc; long n_enc; } pyacc_t;
static int pyacc_init(pyacc_t *P, const char *dir, double um0, const int64_t shape0[3], int shard, int nlev, float q, const char *attrs) {
    memset(P, 0, sizeof *P); P->shard = shard; P->nthreads = 8; pthread_mutex_init(&P->mu, nullptr);
    if (nlev > 8) nlev = 8;
    while (nlev > 1 && shard >> (nlev - 1) < 1) nlev--;
    P->nlev = nlev; P->nz0 = (shape0[0] + shard - 1) / shard; P->mark = 0; P->zdone = 0;
    P->nsub = calloc((size_t)P->nz0 + 1, sizeof(long)); P->nadd = calloc((size_t)P->nz0 + 1, sizeof(long));
    const size_t sv = (size_t)shard * shard * shard; P->poff[0] = 0; P->poff[1] = sv;
    for (int l = 1; l < nlev; l++) {
        pya_level *L = &P->L[l]; L->pe = shard >> l; P->poff[l + 1] = P->poff[l] + (size_t)L->pe * L->pe * L->pe;
        for (int d = 0; d < 3; d++) { L->shape[d] = (shape0[d] + (1 << l) - 1) >> l; L->ns[d] = (L->shape[d] + shard - 1) / shard; }
        char lv[32], d[1600]; z3w_level_name(um0 * (1 << l), lv, sizeof lv); snprintf(d, sizeof d, "%s/%s", dir, lv);
        if (!(L->w = z3w_create(d, L->shape, shard, q, 0, attrs))) { fprintf(stderr, "predict: pyramid: %s\n", z3w_error()); return 1; }
        if (!(L->sh = calloc((size_t)(L->ns[0] * L->ns[1] * L->ns[2]), sizeof *L->sh))) return 1;
    }
    P->nc = z3w_shard_chunks(P->L[1].w);
    return !P->nsub || !P->nadd;
}
static pya_shard *pya_get(pyacc_t *P, pya_level *L, int64_t si) {   /* (mu held) */
    if (!L->sh[si]) { pya_shard *x = calloc(1, sizeof *x); x->enc = calloc((size_t)P->nc, sizeof(uint8_t *)); x->len = calloc((size_t)P->nc, sizeof(size_t)); x->raw = calloc((size_t)P->nc, sizeof(uint8_t *)); L->sh[si] = x; }
    return L->sh[si];
}
/* level l's raw chunks of chunk layer cz (global level-l chunk z) encoded; its shard layers written (mu held) */
static void pya_encode_raw(pyacc_t *P, pya_level *L, pya_shard *x) {
    for (int c = 0; c < P->nc; c++) if (x->raw[c]) {
        if (z3w_encode_chunk(L->w, x->raw[c], &x->enc[c], &x->len[c])) atomic_store(&P->failed, 1);
        free(x->raw[c]); x->raw[c] = nullptr;
    }
}
static void pya_flush(pyacc_t *P, int all) {   /* (mu held) */
    const int64_t zd = all ? INT64_MAX / 4 : P->zdone;   /* level-0 shard layers < zd are complete */
    for (int l = 1; l < P->nlev; l++) {
        pya_level *L = &P->L[l]; const int cg = P->shard / 128;
        if (L->pe < 128)   /* raw chunks whose level-0 layers are all done: encode now (keeps ~one chunk layer raw) */
            for (int64_t ncl = (L->shape[0] + 127) / 128; L->next_clayer < ncl && (all || (((L->next_clayer + 1) * 128) << l) <= zd * P->shard); L->next_clayer++) {
                const int64_t cz = L->next_clayer, Z = cz / cg; const int lz = (int)(cz % cg);
                for (int64_t y = 0; y < L->ns[1]; y++) for (int64_t x = 0; x < L->ns[2]; x++) { pya_shard *s = L->sh[(Z * L->ns[1] + y) * L->ns[2] + x]; if (!s) continue;
                    for (int c = lz * cg * cg; c < (lz + 1) * cg * cg; c++) if (s->raw[c]) { if (z3w_encode_chunk(L->w, s->raw[c], &s->enc[c], &s->len[c])) atomic_store(&P->failed, 1); free(s->raw[c]); s->raw[c] = nullptr; } }
            }
        for (; L->next_layer < L->ns[0] && (all || ((L->next_layer + 1) << l) <= zd); L->next_layer++)
            for (int64_t y = 0; y < L->ns[1]; y++) for (int64_t x = 0; x < L->ns[2]; x++) {
                const int64_t si = (L->next_layer * L->ns[1] + y) * L->ns[2] + x; pya_shard *s = L->sh[si]; if (!s) continue;
                pya_encode_raw(P, L, s);
                if (z3w_write_shard_enc(L->w, L->next_layer, y, x, s->enc, s->len)) { fprintf(stderr, "predict: pyramid: %s\n", z3w_error()); atomic_store(&P->failed, 1); }
                free(s->enc); free(s->len); free(s->raw); free(s); L->sh[si] = nullptr;
            }
    }
}
static void pya_advance(pyacc_t *P) {   /* (mu held) zdone: the first level-0 layer not complete */
    while (P->zdone < P->nz0 && P->zdone < P->mark && P->nadd[P->zdone] == P->nsub[P->zdone]) P->zdone++;
    pya_flush(P, 0);
}
/* main thread: a level-0 shard of layer gz submitted; no later submission has a layer below mark */
static int pyacc_submitted(pyacc_t *P, int64_t gz, int64_t mark) {
    pthread_mutex_lock(&P->mu);
    int bad = gz < P->zdone || gz >= P->nz0;
    if (!bad) P->nsub[gz]++;
    if (mark > P->mark) { P->mark = mark; pya_advance(P); }
    pthread_mutex_unlock(&P->mu);
    if (bad) { fprintf(stderr, "predict: pyramid: shard layer %lld after layer %lld was final (windows out of z order)\n", (long long)gz, (long long)P->zdone); atomic_store(&P->failed, 1); }
    return bad;
}
/* the chunks of the >= 128^3 pieces, encoded by several threads (one writer per head compresses a 512^3 shard with all --threads; the
   pieces' 9 chunks one after another took 27 thread-seconds of 5 heads' 19 Gvox on the pod, on the writers' critical path) */
typedef struct { pyacc_t *P; const uint8_t *buf; const int64_t *g; _Atomic int next; atomic_long ne; } pya_job;
static void *pya_chunk_work(void *a) {
    pya_job *J = a; pyacc_t *P = J->P; const int cg = P->shard / 128; uint8_t *chunk = malloc((size_t)128 * 128 * 128);
    if (!chunk) { atomic_store(&P->failed, 1); return nullptr; }
    for (;;) {   /* item k: level l, chunk c of its piece (levels with pe >= 128 in order) */
        int k = atomic_fetch_add(&J->next, 1), l = 1, m = 0;
        for (; l < P->nlev && P->L[l].pe >= 128; l++) { m = P->L[l].pe / 128; if (k < m * m * m) break; k -= m * m * m; }
        if (l >= P->nlev || P->L[l].pe < 128) break;
        pya_level *L = &P->L[l]; const int pe = L->pe; const uint8_t *pc = J->buf + P->poff[l];
        int64_t S[3], o[3]; for (int d = 0; d < 3; d++) { const int64_t v = J->g[d] * pe; S[d] = v / P->shard; o[d] = v % P->shard; }
        if (S[0] >= L->ns[0] || S[1] >= L->ns[1] || S[2] >= L->ns[2]) continue;
        const int cz = k / (m * m), cy = k / m % m, cx = k % m;
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++)
            memcpy(chunk + ((size_t)z * 128 + y) * 128, pc + (((size_t)(cz * 128 + z) * pe + (cy * 128 + y)) * pe + cx * 128), 128);
        uint8_t *e; size_t n;
        if (z3w_encode_chunk(L->w, chunk, &e, &n)) { atomic_store(&P->failed, 1); continue; }
        atomic_fetch_add(&J->ne, 1);
        const int ci = (int)(((o[0] / 128 + cz) * cg + (o[1] / 128 + cy)) * cg + (o[2] / 128 + cx));
        const int64_t si = (S[0] * L->ns[1] + S[1]) * L->ns[2] + S[2];
        pthread_mutex_lock(&P->mu); pya_shard *x = pya_get(P, L, si); free(x->enc[ci]); x->enc[ci] = e; x->len[ci] = n; pthread_mutex_unlock(&P->mu);
    }
    free(chunk); return nullptr;
}
/* writer thread: the pieces of level-0 shard (gz, gy, gx) at buf + poff[l] */
static void pyacc_post(void *ctx, const uint8_t *buf, int64_t gz, int64_t gy, int64_t gx) {
    pyacc_t *P = ctx; const int64_t g[3] = {gz, gy, gx}; const int cg = P->shard / 128;
    const double t0 = now(); long ne = 0;
    {   int nitem = 0; for (int l = 1; l < P->nlev && P->L[l].pe >= 128; l++) { const int m = P->L[l].pe / 128; nitem += m * m * m; }
        if (nitem) { pya_job J = {.P = P, .buf = buf, .g = g}; pthread_t th[16]; int nt = nitem < P->nthreads ? nitem : P->nthreads; if (nt > 16) nt = 16; if (nt < 1) nt = 1;
                     for (int i = 1; i < nt; i++) pthread_create(&th[i], nullptr, pya_chunk_work, &J);
                     pya_chunk_work(&J); for (int i = 1; i < nt; i++) pthread_join(th[i], nullptr); ne = atomic_load(&J.ne); } }
    for (int l = 1; l < P->nlev; l++) {
        if (P->L[l].pe >= 128) continue;   /* (encoded above) */
        pya_level *L = &P->L[l]; const int pe = L->pe; const uint8_t *pc = buf + P->poff[l];
        int64_t S[3], o[3];   /* the level-l shard and the piece's offset in it */
        for (int d = 0; d < 3; d++) { const int64_t v = g[d] * pe; S[d] = v / P->shard; o[d] = v % P->shard; }
        if (S[0] >= L->ns[0] || S[1] >= L->ns[1] || S[2] >= L->ns[2]) continue;
        const int64_t si = (S[0] * L->ns[1] + S[1]) * L->ns[2] + S[2];
        {
            int any = 0; const size_t pv = (size_t)pe * pe * pe; for (size_t i = 0; i < pv && !any; i++) any = pc[i] != 0;
            if (!any) continue;   /* (a chunk never touched stays fill) */
            const int ci = (int)(((o[0] / 128) * cg + o[1] / 128) * cg + o[2] / 128); const int64_t r[3] = {o[0] % 128, o[1] % 128, o[2] % 128};
            pthread_mutex_lock(&P->mu); pya_shard *x = pya_get(P, L, si);
            if (!x->raw[ci] && !(x->raw[ci] = calloc(((size_t)128 * 128 * 128), 1))) atomic_store(&P->failed, 1);
            else for (int z = 0; z < pe; z++) for (int y = 0; y < pe; y++) memcpy(x->raw[ci] + (((size_t)(r[0] + z) * 128 + (r[1] + y)) * 128 + r[2]), pc + ((size_t)z * pe + y) * pe, (size_t)pe);
            pthread_mutex_unlock(&P->mu);
        }
    }
    pthread_mutex_lock(&P->mu); P->nadd[gz]++; P->t_enc += now() - t0; P->n_enc += ne; pya_advance(P); pthread_mutex_unlock(&P->mu);
}
static int pyacc_finish(pyacc_t *P) {   /* after the writers finished: everything left */
    pthread_mutex_lock(&P->mu); pya_flush(P, 1); pthread_mutex_unlock(&P->mu);
    for (int l = 1; l < P->nlev; l++) { z3w_close(P->L[l].w); free(P->L[l].sh); }
    free(P->nsub); free(P->nadd);
    return atomic_load(&P->failed);
}
typedef struct { char name[40], content[160], attrs[1400], dir[1400]; int c0, c1, sb0, sb1, s0, s1; writer_t *wr; z3w *w; uint8_t *dsh2[2]; void *gv[2]; int sb; pyacc_t *pya;
                 const float *vrs; const uint16_t *vsc; int vfma; size_t vhdr; } phead_t;   /* vrs: the GPU encoder's tables (nullptr: the CPU encodes from the bytes); vhdr: the levels' header bytes */
/* the heads' device shards are double-buffered (dsh2[b], b toggling per shard): the copy of one shard to the host runs on the offload stream
   while the next shard is placed; gv[b] the copy that last read buffer b (the compute stream waits for it before reuse) */
static void phead_d2h(phead_t *h, int b) {
    size_t n = h->wr->bytes;
    if (h->pya) { pyacc_t *P = h->pya; for (int l = 1; l < P->nlev; l++) nn_pool2_u8(h->dsh2[b] + P->poff[l - 1], P->shard >> (l - 1), h->dsh2[b] + P->poff[l]); n = P->poff[P->nlev]; }
    if (h->vrs) { writer_t *W = h->wr; vcg_layout L; vcg_layout_of(&L, W->vn, W->vcap); uint8_t *hb = W->buf[h->sb] + W->voff;
                  nn_vcg_shard(h->dsh2[b], W->vn, h->vrs, h->vsc, h->vfma, h->dsh2[b] + W->voff, (int32_t *)(hb + L.o_lv), (uint16_t *)(hb + L.o_pos), W->vcap);   /* (the levels straight into the pinned slot) */
                  n = W->voff + h->vhdr; }
    if (!h->wr->async) { nn_d2h(h->wr->buf[h->sb], h->dsh2[b], n); h->gv[b] = nullptr; return; }
    nn_offload_copy(h->wr->buf[h->sb], h->dsh2[b], n, 1, h->wr->evs[h->sb], h->wr->evd[h->sb]); h->gv[b] = h->wr->evd[h->sb];
}
/* a head's finished device shard (b) to its writer: the pyramid pieces, the copy, the submission; mark: no later shard below layer mark */
static int phead_submit(phead_t *h, int b, int64_t sz, int64_t sy, int64_t sx, int64_t mark) {
    phead_d2h(h, b);
    if (h->pya && pyacc_submitted(h->pya, sz + h->wr->off[0], mark + h->wr->off[0])) return 1;
    if (writer_submit(h->wr, h->sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    h->sb = writer_next(h->wr, h->sb);
    return 0;
}
static void phead_fresh(phead_t *h, int b) { if (h->gv[b]) nn_stream_wait(0, h->gv[b]); h->gv[b] = nullptr; nn_zero(h->dsh2[b], h->wr->bytes); }
/* the fine stem's output row of channel c (train: the voxel-logit outputs c < 2 or c >= 9 below cout, the first fstem_out of them; -1 none) */
static int phead_stem_row(const unet_cfg *cfg, int c) {
    if (!cfg->fstem) return -1;
    int k = 0; const int nb = cfg->lat_guide == 64 ? cfg->cout - 128 : cfg->cout;
    for (int j = 0; j < nb && k < cfg->fstem_out; j++) if (j < 2 || j >= 9) { if (j == c) return k; k++; }
    return -1;
}
static int phead_add(phead_t *H, int *nh, const unet_cfg *cfg, const char *name, int c0, int c1, const char *content) {
    const int nb = cfg->lat_guide == 64 ? cfg->cout - 128 : cfg->cout;
    if (*nh >= PHEAD_MAX || c0 < 0 || c0 >= nb || c1 >= nb) { fprintf(stderr, "predict --heads: %s: channel %d / %d outside the %d head outputs (or more than %d heads)\n", name, c0, c1, nb, PHEAD_MAX); return 1; }
    phead_t *h = &H[(*nh)++]; memset(h, 0, sizeof *h);
    snprintf(h->name, sizeof h->name, "%s", name); snprintf(h->content, sizeof h->content, "%s", content);
    h->c0 = c0; h->c1 = c1;
    h->sb0 = cfg->lat_guide == 64 && c0 < 2 ? cfg->cout - 128 + 64 * c0 : -1; h->sb1 = cfg->lat_guide == 64 && c1 >= 0 && c1 < 2 ? cfg->cout - 128 + 64 * c1 : -1;
    h->s0 = phead_stem_row(cfg, c0); h->s1 = c1 >= 0 ? phead_stem_row(cfg, c1) : -1;
    return 0;
}
static int hdr_int(const char *h, const char *key, int dflt) { const char *q = h ? strstr(h, key) : nullptr; return q ? atoi(q + strlen(key)) : dflt; }
/* --heads LIST: rv (the recto / verso pair, fused), ink, fib (the fibre head: fib/H, fib/V, fib/dyad/c2 .. c7 as fiber-trace --mask / --dyad
   read them, fib/fam), cN (channel N's probability), all (every one the checkpoint has) */
static int phead_parse(const char *spec, const char *hdr, const unet_cfg *cfg, int rv2c, phead_t *H, int *nh) {
    char *t = strdup(spec); int bad = 0;
    for (char *q = strtok(t, ","); q && !bad; q = strtok(nullptr, ",")) {
        const int all = !strcmp(q, "all");
        const int isrv1 = hdr && strstr(hdr, "\"rv\":{\"version\":1"), ink = hdr_int(hdr, "\"ink\":", -1);
        const char *fb = hdr ? strstr(hdr, "\"fib_dir\":{\"version\":1,\"head\":1,\"channels\":{") : nullptr;
        if (!strcmp(q, "rv") || (all && (rv2c >= 0 || isrv1))) {
            if (rv2c < 0 && !isrv1) { fprintf(stderr, "predict --heads rv: the checkpoint has no recto / verso head (train --rv)\n"); bad = 1; break; }
            bad |= phead_add(H, nh, cfg, "rv", rv2c >= 0 ? rv2c : 0, rv2c >= 0 ? rv2c + 1 : 1, "recto / verso side: 1 + 254 (1 + p_recto - p_verso) / 2 (0: CT 0)");
        }
        if (!strcmp(q, "ink") || (all && ink >= 0)) {
            if (ink < 0) { fprintf(stderr, "predict --heads ink: the checkpoint has no ink head (train --ink-head)\n"); bad = 1; break; }
            char c[96]; snprintf(c, sizeof c, "ink probability (unwrap2 channel %d)", ink); bad |= phead_add(H, nh, cfg, "ink", ink, -1, c);
        }
        if (!strcmp(q, "fib") || (all && fb)) {
            if (!fb) { fprintf(stderr, "predict --heads fib: the checkpoint has no fibre head (train --fib-head)\n"); bad = 1; break; }
            const int ch = hdr_int(fb, "\"H\":", -1), cv = hdr_int(fb, "\"V\":", -1), fam = hdr_int(fb, "\"family_v\":", -1); const char *dy = strstr(fb, "\"dyad\":[");
            char c[96]; snprintf(c, sizeof c, "fibre H probability (channel %d)", ch); bad |= phead_add(H, nh, cfg, "fib/H", ch, -1, c);
            snprintf(c, sizeof c, "fibre V probability (channel %d)", cv); bad |= phead_add(H, nh, cfg, "fib/V", cv, -1, c);
            for (int k = 0; k < 6 && dy && !bad; k++) {   /* (named c2 .. c7: fiber-trace --dyad reads DIR/c2 .. DIR/c7) */
                int dc = -1; const char *p = dy + 8; for (int j = 0; j < k && p; j++) { p = strchr(p, ','); if (p) p++; } if (p) dc = atoi(p);
                char n[40]; snprintf(n, sizeof n, "fib/dyad/c%d", 2 + k); snprintf(c, sizeof c, "fibre axis dyad entry %d (channel %d; zz yy xx 0.5+zy 0.5+zx 0.5+yx)", k, dc);
                bad |= phead_add(H, nh, cfg, n, dc, -1, c);
            }
            if (fam >= 0 && !bad) { snprintf(c, sizeof c, "fibre family V probability (channel %d)", fam); bad |= phead_add(H, nh, cfg, "fib/fam", fam, -1, c); }
        }
        if (q[0] == 'c' && q[1] >= '0' && q[1] <= '9') { char c[96]; snprintf(c, sizeof c, "probability of channel %d", atoi(q + 1)); bad |= phead_add(H, nh, cfg, q, atoi(q + 1), -1, c); }
        else if (!all && strcmp(q, "rv") && strcmp(q, "ink") && strcmp(q, "fib")) { fprintf(stderr, "predict --heads: unknown head %s (rv, ink, fib, cN, all)\n", q); bad = 1; }
    }
    free(t);
    if (!bad && !*nh) { fprintf(stderr, "predict --heads %s: no head\n", spec); bad = 1; }
    return bad;
}
typedef struct { phead_t *h; double um; const int64_t *ash; int shard, nlev, nthreads, fail; float q; } phead_pyr;
static void *phead_pyr_main(void *a) {
    phead_pyr *p = a;
    for (int l = 1; l < p->nlev && !p->fail; l++) if (pyramid_build_level_z(p->h->dir, p->um, l, p->ash, p->shard, p->q, 0, p->nthreads, p->h->attrs, 0, -1, 1)) p->fail = 1;
    if (!p->fail && pyramid_write_group(p->h->dir, p->um, p->nlev, "ufsm-recto", p->h->attrs)) p->fail = 1;
    if (p->fail) fprintf(stderr, "predict: head %s: the pyramid failed\n", p->h->name);
    return nullptr;
}
/* a head's writer: threads and events as the main writer's, nb pinned shard buffers */
static int phead_writer(writer_t *W, z3w *w, size_t bytes, size_t abytes, int nthreads, int nb, int gpu, int async) {
    memset(W, 0, sizeof *W); W->w = w; W->nthreads = nthreads; W->bytes = bytes; W->donefd = -1; W->gpu = gpu; W->async = async;
    if (nb < 2) nb = 2;
    if (nb > WBUF_MAX) nb = WBUF_MAX;
    for (int k = 0; k < nb; k++) if (!(W->buf[k] = nn_host_alloc_try(abytes))) { if (k < 2) return 1; nb = k; break; }
    W->nb = nb;
    pthread_mutex_init(&W->mu, nullptr); pthread_cond_init(&W->cv, nullptr);
    if (async) for (int k = 0; k < WBUF_MAX; k++) { W->evs[k] = nn_event_create(); W->evd[k] = nn_event_create(); }
    const int nc = (int)((bytes / ((size_t)128 * 128 * 128)) > 0 ? bytes / ((size_t)128 * 128 * 128) : 1), nt = nthreads > 0 ? nthreads : (int)ufsm_ncpu();
    W->nw = nt / nc; if (W->nw < 1) W->nw = 1; if (W->nw > WRITERS_MAX) W->nw = WRITERS_MAX;
    W->wt = nt / W->nw > 0 ? nt / W->nw : 1;
    for (int i = 0; i < W->nw; i++) pthread_create(&W->wth[i], nullptr, writer_main, W);
    return 0;
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
    double norm_mean, norm_sd;   /* --norm: fixed input z-score when norm_sd > 0 */
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
        const double w3 = (double)W * W * W, wmean0 = h[1] / w3, var = h[2] / w3 - wmean0 * wmean0;
        const double wmean = J->norm_sd > 0 ? J->norm_mean : wmean0, sd = J->norm_sd > 0 ? J->norm_sd : sqrt(var > 0 ? var : 0) + 1e-3;
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

/* UFSM_PRED_CELL_Q (measurement for a cell-output prediction store): the latent-grid head's cell logits [C][d][h][w] of each window
   rounded in place to a per-channel step (one value for all channels or C comma-separated; 0 = statistics only, no rounding), int8
   indices clamped to +-127; over each window's interior cells (halo cells off) per-channel mean / sd / |max|, index entropy, clipped
   count and the zstd-3 size of the int8 planes, reported at exit as bytes per voxel (64 voxels per cell). */
static struct { int on, C; float st[256]; double n, s[256], s2[256], amax[256], zb[256], clip[256]; uint64_t hist[256][255]; void *h; size_t hn; } g_cq;
static void cellq_report(void) {
    if (!g_cq.n) return;
    double tz = 0, te = 0; fprintf(stderr, "cellq: %.0f interior cells (%.3g Gvox), C %d\n", g_cq.n, g_cq.n * 64e-9, g_cq.C);
    for (int c = 0; c < g_cq.C; c++) {
        const double m = g_cq.s[c] / g_cq.n, sd = sqrt(fmax(0, g_cq.s2[c] / g_cq.n - m * m)); double e = 0;
        for (int k = 0; k < 255; k++) if (g_cq.hist[c][k]) { const double p = g_cq.hist[c][k] / g_cq.n; e -= p * log2(p); }
        tz += g_cq.zb[c]; te += e / 8;
        fprintf(stderr, "cellq: ch %2d mean %+8.4f sd %7.4f |max| %7.3f step %.4g  H %.3f bit  zstd %.4f B/cell  clipped %.2g\n", c, m, sd, g_cq.amax[c], g_cq.st[c],
                g_cq.st[c] > 0 ? e : 0., g_cq.zb[c] / g_cq.n, g_cq.clip[c] / g_cq.n);
    }
    if (g_cq.st[0] > 0) fprintf(stderr, "cellq: total %.4f B/voxel zstd-3 (H0 %.4f B/voxel), %.3f B/cell\n", tz / (g_cq.n * 64), te / 64, tz / g_cq.n);
}
static void cellq_init(int C) {
    const char *e = getenv("UFSM_PRED_CELL_Q"); g_cq.on = 1; g_cq.C = C < 256 ? C : 256; int k = 0;
    for (const char *p = e; *p && k < g_cq.C; k++) { g_cq.st[k] = strtof(p, (char **)&p); if (*p == ',') p++; else break; }
    for (int c = k > 0 ? k : 1; c < g_cq.C; c++) g_cq.st[c] = g_cq.st[k > 0 ? k - 1 : 0];   /* (the last value repeats) */
    atexit(cellq_report);
}
static void cellq_apply(const void *ld, int lh16, int C, int d, int h, int w, int hz, int hxy) {   /* hz, hxy: halo in cells */
    if (!g_cq.on) cellq_init(C);
    const size_t n = (size_t)C * d * h * w, es = lh16 ? 2 : 4;
    if (g_cq.hn < n * es) { free(g_cq.h); g_cq.h = malloc(n * es); g_cq.hn = n * es; }
    nn_sync(); nn_d2h(g_cq.h, ld, n * es);
    const int iz = d - 2 * hz, iy = h - 2 * hxy, ix = w - 2 * hxy; if (iz <= 0 || iy <= 0 || ix <= 0) return;
    const size_t ni = (size_t)iz * iy * ix; int8_t *q = malloc(ni); const size_t zc = ZSTD_compressBound(ni); void *zb = malloc(zc);
    for (int c = 0; c < g_cq.C; c++) {
        const float st = g_cq.st[c]; size_t j = 0;
        for (int z = 0; z < d; z++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            const size_t i = (((size_t)c * d + z) * h + y) * w + x;
            float v = lh16 ? (float)((_Float16 *)g_cq.h)[i] : ((float *)g_cq.h)[i];
            int qi = 0;
            if (st > 0) { const float r = rintf(v / st); qi = r > 127 ? 127 : r < -127 ? -127 : (int)r; v = qi * st; }
            if (lh16) ((_Float16 *)g_cq.h)[i] = (_Float16)v; else ((float *)g_cq.h)[i] = v;
            if (z < hz || z >= d - hz || y < hxy || y >= h - hxy || x < hxy || x >= w - hxy) continue;
            const float o = lh16 ? (float)((_Float16 *)g_cq.h)[i] : v;
            g_cq.s[c] += o; g_cq.s2[c] += (double)o * o; if (fabs(o) > g_cq.amax[c]) g_cq.amax[c] = fabs(o);
            if (st > 0) { g_cq.hist[c][qi + 127]++; g_cq.clip[c] += qi == 127 || qi == -127; q[j++] = (int8_t)qi; }
        }
        if (st > 0) g_cq.zb[c] += (double)ZSTD_compress(zb, zc, q, ni, 3);
    }
    g_cq.n += (double)ni; free(q); free(zb);
    if (g_cq.st[0] > 0) nn_h2d((void *)ld, g_cq.h, n * es);
}

int cmd_predict(int argc, char **argv) {
    /* UFSM_PRED_DEVREAD=1: the CT windows decoded on the GPU into device memory (volcomp's GPU decoder, byte-identical): turned on here,
       before the first z3 read opens the decoder, on the predict GPU */
    const int devread = ufsm_env_on("UFSM_PRED_DEVREAD");
    if (devread) { setenv("UFSM_GPU_DECODE", "1", 0); if (!getenv("UFSM_GPU_DECODE_DEV")) setenv("UFSM_GPU_DECODE_DEV", opt(argc, argv, "--gpu", "0"), 1); }
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
                        "       [--channel 0]   output channel written (band_affinity checkpoints: 1..6 = affinities z1 y1 x1 z8 y8 x8; ink: a task unwrap2 --ink-head\n"
                        "                       checkpoint's ink channel; with --fit-zarr / --fit-fields the volume of that channel beside the fields, one pass)\n"
                        "       [--um-target U]   instead of --level: the level nearest U um (e.g. 9.6: the universal net's operating rung on any scan)\n"
                        "       [--heads LIST]   latent-grid (train --lat 4) checkpoints: every listed head from one trunk pass, each its own store <out>/<name>:\n"
                        "                        rv (recto / verso fused), ink, fib (fib/H, fib/V, fib/dyad/c2..c7, fib/fam), cN (channel N), all; with --fit-zarr too\n"
                        "       [--fit-fields DIR] [--fit-h 8]   task unwrap checkpoints: also the winding fitter's input fields of the box (ufsm fit DIR)\n"
                        "       [--fit-zarr DIR]                 the same fields as zarr arrays, flushed slab by slab (whole volumes)\n"
                        "       [--tta K]   the fit fields, or without --fit-*: the probability, averaged over K window symmetries (z / y / x flips, y <-> x swap; 1..12, src/tta.h)\n"
                        "       [--view-level L] [--view-map REG.json]   scan-fusion checkpoints (view_P): the view crop per window from level L of this scan\n"
                        "                   or, with a map, the registered partner scan on this scan's level-L grid (use the training window size)\n"
                        "       [--fit-w4-d 32]   task unwrap2 checkpoints: the winding mod 4 relative planes dz dy dx dq over this many voxels (<= the halos)\n"
                        "       [--fit-ct 1]                     with them the CT planes ct / cs (fit --w-ct / --w-cspacing; ufsm ct-fields adds them later)\n"
                        "       [--no-out 1]                     skip writing the prediction volume's shards (e.g. only the --fit-zarr fields)\n"
                        "       [--fit-q 8] [--fit-q-phase 4]   volcomp quantiser of the --fit-zarr arrays (0 lossless; the phase planes pc / ps: --fit-q-phase)\n"
                        "       [--norm MEAN,SD]   fixed input normalization (uint8 CT units) instead of each window's own mean / sd\n"
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
    /* --um-target U (instead of --level): the pyramid level whose voxel size is nearest U um in octaves (a universal checkpoint at its 9.6 um
       rung over any scan: 2.4 um -> level 2, 7.91 -> 0, 4.8 -> 1); the scale FiLM codes follow the level read */
    if (*opt(argc, argv, "--um-target", "") && !*opt(argc, argv, "--level", "")) {
        const double ut = atof(opt(argc, argv, "--um-target", "0")), u0 = atof(opt(argc, argv, "--um", "0"));
        if (!(ut > 0) || !(u0 > 0)) { fprintf(stderr, "predict: --um-target needs a positive target and --um\n"); return 2; }
        store *s0 = store_open(argv[3]); int best = -1; double bd = 1e30;
        for (int l = 0; s0 && l < 12; l++) { z3 *z = pyramid_open_level(s0, argv[4], l, u0, opt(argc, argv, "--cache", nullptr)); if (!z) continue;
            const double d = fabs(log2(u0 * (1 << l) / ut)); if (d < bd - 1e-9) { bd = d; best = l; } z3_close(z); }
        if (s0) store_close(s0);
        if (best < 0) { fprintf(stderr, "predict: --um-target: no level of %s opens\n", argv[4]); return 1; }
        level = best; fprintf(stderr, "predict: --um-target %g: level %d (%g um, %.2f octaves off)\n", ut, level, u0 * (1 << level), bd);
    }
    int shard = atoi(opt(argc, argv, "--shard", "512")), gpu = atoi(opt(argc, argv, "--gpu", "0")), nlev = atoi(opt(argc, argv, "--levels", "4"));
    double norm_mean = 0, norm_sd = 0;   /* --norm MEAN,SD: fixed input normalization (uint8 CT units) instead of each window's own mean / sd */
    { const char *nm = opt(argc, argv, "--norm", nullptr);
      if (nm && (sscanf(nm, "%lf,%lf", &norm_mean, &norm_sd) != 2 || !(norm_sd > 0))) { fprintf(stderr, "predict: --norm needs MEAN,SD with SD > 0\n"); return 2; } }
    const double input_gain = atof(opt(argc, argv, "--input-gain", "1")), input_bias = atof(opt(argc, argv, "--input-bias", "0"));
    if (!isfinite(input_gain) || input_gain <= 0 || !isfinite(input_bias)) { fprintf(stderr, "predict: --input-gain must be finite and positive; --input-bias finite\n"); return 2; }
    const int input_affine = input_gain != 1 || input_bias != 0;
    char input_attrs[128] = "";
    if (input_affine) snprintf(input_attrs, sizeof input_attrs, ",\"input_gain\":%.9g,\"input_bias\":%.9g", input_gain, input_bias);
    const int win_given = *opt(argc, argv, "--window", "") || *opt(argc, argv, "--halo", "");
    /* --window auto (or UFSM_PRED_WINDOW=auto; the production scripts pass it): the cheapest window per interior voxel this GPU holds (see
       below): bigger windows on big GPUs, today's on 16 GB cards. Windows other than the production ones change the outputs (per-window
       GroupNorm statistics, context); allowed without a gate by the user 10-31 (scripts/pod/wingate.sh is the gate, not run).
       UFSM_PRED_WINDOW=prod: the production window (fields 384 x 640^2 halo 64, --feather ink 576 halo 32) */
    const char *wenv = getenv("UFSM_PRED_WINDOW");
    const int wauto = !strcmp(opt(argc, argv, "--window", ""), "auto") || (wenv && (!strcmp(wenv, "auto") || !strcmp(wenv, "prod")));
    const int wprod = wauto && wenv && !strcmp(wenv, "prod");
    if (wauto) { W = 640; if (!*opt(argc, argv, "--halo", "")) halo = 64; }
    /* --window-z WZ [--halo-z HZ]: slab windows WZ x W x W (train --Pz nets; HZ default WZ / 4), the plain GPU path only */
    int Wz = atoi(opt(argc, argv, "--window-z", "0")), hz = atoi(opt(argc, argv, "--halo-z", "-1"));
    int slab = Wz > 0 && Wz != W;
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
    const char *chs = opt(argc, argv, "--channel", "0");   /* output channel to write (band_affinity: 1..6 = affinities; "ink": a task unwrap2 --ink-head checkpoint's ink channel) */
    /* --channel rv (train --task unwrap2 --rv checkpoints): both side channels in one pass, written as 1 + 254 (1 + p_recto - p_verso) / 2, 0 where CT 0 */
    const int rvfuse = !strcmp(chs, "rv");
    /* --latents STORE (ufsm lc-encode): a --lat-basis checkpoint's cell channels come from the stored quantised coefficients (the window's
       mean / sd from them too), not from the CT window, which is still read for the air mask and the radial channels */
    const lv_codec *lvc = nullptr; int lvm = 0, lvq = 0, lv_stride = 0; nn_lv *lvL = nullptr;   /* the CT level is a latvol array: see lvm below */
    const char *lat_store = opt(argc, argv, "--latents", nullptr); zr lzr; const float *lz_bas = nullptr; double lz_sraw = 0, lz_drop = 0; long long lz_org[3] = {0, 0, 0}; int lz_c = 0; void *lz_d = nullptr;
    if (lat_store) {
        char zp[1200]; snprintf(zp, sizeof zp, "%s/zarr.json", lat_store); FILE *zf = fopen(zp, "r"); static char zt[65536]; size_t zn = zf ? fread(zt, 1, sizeof zt - 1, zf) : 0; if (zf) fclose(zf); zt[zn] = 0;
        const char *q1 = strstr(zt, "\"step_raw\":"), *q2 = strstr(zt, "\"origin\":["), *q3 = strstr(zt, "\"lat_c\":");
        if (!q1 || !q2 || !q3 || sscanf(q2 + 10, "%lld,%lld,%lld", lz_org, lz_org + 1, lz_org + 2) != 3 || zr_open(&lzr, lat_store)) { fprintf(stderr, "predict: --latents %s is not a latcomp store (ufsm lc-encode)\n", lat_store); return 2; }
        lz_sraw = atof(q1 + 11); lz_c = atoi(q3 + 8);
        { const char *q4 = strstr(zt, "\"drop_ms\":"); lz_drop = q4 ? atof(q4 + 10) : 0; }   /* (stores before the attribute: no correction) */
        if (degrade || opt(argc, argv, "--aux", nullptr) || !strcmp(opt(argc, argv, "--split", "0"), "z") || atoi(opt(argc, argv, "--tta", "1")) > 1) {
            fprintf(stderr, "predict: --latents: no --aug-*, --aux, --split or --tta (they read the CT itself)\n"); return 2; }   /* (--fit-ct: the rebuilt CT) */
    }
    const int fstem_off = atoi(opt(argc, argv, "--fstem-off", "0"));   /* --fstem-off 1: a train --fstem checkpoint without its fine stem (the ablation; the trunk's fast paths stay off) */
    int rv2c = -1;   /* train --rv 2 checkpoints: the recto / verso pair on its own channels (rv2c, rv2c + 1), interpolated (never the guided / sub-cell pair) */
    { char *h = checkpoint_header_line(ckpt); const int isrv = h && strstr(h, "\"rv\":{\"version\":1"); const char *r2 = h ? strstr(h, "\"rv\":{\"version\":2,\"recto\":") : nullptr;
      if (r2) rv2c = atoi(r2 + strlen("\"rv\":{\"version\":2,\"recto\":"));
      free(h);
      if (rvfuse && !isrv && rv2c < 0) { fprintf(stderr, "predict: --channel rv: %s is not a train --rv checkpoint\n", ckpt); return 2; }
      if (isrv && (opt(argc, argv, "--fit-fields", nullptr) || opt(argc, argv, "--fit-zarr", nullptr))) { fprintf(stderr, "predict: %s is a recto / verso (train --rv) checkpoint: channels 0-1 are not the phase (no --fit-*)\n", ckpt); return 2; } }
    const int out_ch = !strcmp(chs, "ink") ? checkpoint_ink_channel(ckpt) : rvfuse ? 0 : atoi(chs);
    if (!strcmp(chs, "ink") && out_ch < 0) { fprintf(stderr, "predict: --channel ink: %s has no ink channel (train --task unwrap2 --ink-head 1)\n", ckpt); return 2; }
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
    /* --tta K: the fit fields averaged over K window symmetries (src/tta.h; identity first; the probability output is then the last pass);
       without --fit-*: the probability output itself is the mean of the K passes' probabilities (fp16 accumulator, 2 B per window voxel) */
    const int tta = atoi(opt(argc, argv, "--tta", "1"));
    const int fea = atoi(opt(argc, argv, "--feather", "0"));   /* --feather B (above) */
    if (tta < 1 || tta > TTA_MAX) { fprintf(stderr, "predict: --tta 1..%d\n", TTA_MAX); return 2; }
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
    { FILE *cf=fopen(ckpt,"rb"); if (!cf) { fprintf(stderr,"predict: cannot open checkpoint %s: %s\n",ckpt,strerror(errno)); return 1; } fclose(cf); }   /* (sheet_checkpoint returns -1 for a missing file too, which read as a contract error) */
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
    if (ffd && (cfg.cout < 11 || (half && half != 3) || sheet_task)) { fprintf(stderr, "predict: --fit-fields needs a task unwrap checkpoint (11+ outputs) at full resolution or on the latent grid (train --lat 4)\n"); return 2; }
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
    if (!strcmp(chs, "ink")) snprintf(content, sizeof content, "ink probability (unwrap2 channel %d)", out_ch);
    if (rvfuse) { if ((half && half != 3) || hmode == 2) { fprintf(stderr, "predict: --channel rv: full-resolution or latent-grid (train --lat 4) checkpoints only\n"); return 2; }
                  snprintf(content, sizeof content, "recto / verso side: 1 + 254 (1 + p_recto - p_verso) / 2 (0: CT 0)"); }
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
    /* the multi-head path (phead_t above): --heads LIST, or the single output of a latent-grid net whose fast paths do not cover it (a fine
       stem: the fast paths read no CT; --rv 2's own pair; a plain channel of the sub-cell head, which the pool4 chain gave without its
       sub-cell term). UFSM_PRED_HEADS=0: the previous paths for the single output */
    const char *hspec = opt(argc, argv, "--heads", nullptr);
    phead_t PH[PHEAD_MAX]; int nph = 0;
    const int lat_ok = half == 3 && hdiv == 4 && hmode == 1 && !ufsm_env_on("UFSM_PRED_HOSTPATH") && !fea && tta <= 1 && !split && !sheet_task && !slab && !sweep_out && nwork == 1 && cfg.cin != 5;
    if (hspec) {
        if (!lat_ok) { fprintf(stderr, "predict --heads: latent-grid (train --lat 4) checkpoints on one GPU's device path (no --feather, --tta, --split, --window-z, --gpus workers)\n"); return 2; }
        char *h = checkpoint_header_line(ckpt); const int bad = phead_parse(hspec, h, &cfg, rv2c, PH, &nph); free(h);
        if (bad) return 2;
    } else if (lat_ok && (cfg.fstem || rv2c >= 0 || (cfg.lat_guide == 64 && !rvfuse && out_ch < 2)) && !atoi(opt(argc, argv, "--no-out", "0")) && (!getenv("UFSM_PRED_HEADS") || ufsm_env_on("UFSM_PRED_HEADS"))) {
        if (rvfuse) { if (phead_add(PH, &nph, &cfg, "rv", rv2c >= 0 ? rv2c : 0, rv2c >= 0 ? rv2c + 1 : 1, content)) return 2; }
        else if (phead_add(PH, &nph, &cfg, chs, out_ch, -1, content)) return 2;
    }
    const int lhm = nph > 0;   /* every output from the cell logits per shard (no full-resolution logits) */
    int ph_c0[PHEAD_MAX], ph_c1[PHEAD_MAX], ph_sb0[PHEAD_MAX], ph_sb1[PHEAD_MAX], ph_s0[PHEAD_MAX], ph_s1[PHEAD_MAX], phd = 0;   /* (nn_lat_heads_place; phd: the shard buffer in use) */
    for (int k = 0; k < nph; k++) { ph_c0[k] = PH[k].c0; ph_c1[k] = PH[k].c1; ph_sb0[k] = PH[k].sb0; ph_sb1[k] = PH[k].sb1; ph_s0[k] = PH[k].s0; ph_s1[k] = PH[k].s1; }
    if (lhm) { fprintf(stderr, "predict: %d head%s from one trunk pass:", nph, nph > 1 ? "s" : ""); for (int k = 0; k < nph; k++) { char c1[16] = ""; if (PH[k].c1 >= 0) snprintf(c1, sizeof c1, "+%d", PH[k].c1);
                                     fprintf(stderr, " %s (ch %d%s%s%s)", PH[k].name, PH[k].c0, c1, PH[k].sb0 >= 0 ? " sub-cell" : "", PH[k].s0 >= 0 ? " stem" : ""); } fprintf(stderr, "\n"); }
    if (!ffd && tta > 1 && (split || half || sheet_task || rvfuse)) { fprintf(stderr, "predict: probability --tta: not with --split, half-resolution, sheet or recto/verso-fused outputs\n"); return 2; }
    if (ffd && out_ch && tta > 1) { fprintf(stderr, "predict: --channel N with --fit-* and --tta: not supported (the probability volume would come from the last symmetry pass)\n"); return 2; }
    if (ffd && (split || ngpu > 1)) { fprintf(stderr, "predict: --fit-fields runs on one GPU\n"); return 2; }   /* (--channel N: the probability volume of channel N beside the fields, e.g. --channel ink) */
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
        if (ffd || sweep_out || lhm) unet_set_head_channels(uk, 0, cfg.cout);   /* (--heads: every channel on the cell grid) */   /* every head: the recto channel 0 is written, the others pooled (or swept) */
        else if (rvfuse) unet_set_head_channels(uk, rv2c >= 0 && !cfg.lat_guide ? rv2c : 0, cfg.lat_guide ? cfg.cout : 2);   /* --channel rv: both side logits, fused into channel 0 after the forward */
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
        if (cfg.film) {   /* per-level scale FiLM (train --film): codes log2(um / 4.8) of the grid and of the data, both the CT level's voxel size (no zoom);
                             --scale-um / --film-data-um override them (e.g. an upsampled coarse scan: data um its native size) */
            const double g_ = atof(opt(argc, argv, "--scale-um", "0")), d_ = atof(opt(argc, argv, "--film-data-um", "0")), lu = um * (1 << level);
            const float c[2] = {unet_scale_code(g_ > 0 ? g_ : lu), unet_scale_code(d_ > 0 ? d_ : g_ > 0 ? g_ : lu)};
            unet_set_film(uk, c, cfg.film);
            if (!k) fprintf(stderr, "predict: scale FiLM codes %.3f / %.3f (grid / data um %g / %g)\n", c[0], c[1], g_ > 0 ? g_ : lu, d_ > 0 ? d_ : g_ > 0 ? g_ : lu);
        }
        if (k) nn_init(gpu);
    }
    free(gnh);
    store *s = store_open(root);
    char ak[1200];
    snprintf(ak, sizeof ak, "%s/level%d", key, level);
    z3 *ct = pyramid_open_level(s, key, level, um, cache);
    lvc = ct ? z3_latvol(ct) : nullptr;
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
    if (wauto) {
        if (split || half || sheet_task || sweep_out || ngpu > 1 || ufsm_env_on("UFSM_PRED_HOSTPATH")) { fprintf(stderr, "predict: --window auto: one GPU's plain CT path only\n"); return 2; }
        /* candidates (z, xy, halo), production first; cost = computed voxels per interior voxel, x 1.25 (fields) / 1.22 (ink) without the
           direct slabs (desk measurements); the cheapest whose dry-run memory (+ CT window, field accumulators, feather E, 1.5 GB) is free wins.
           Fields (--fit-zarr strides with 512 shards: per-shard z 256 / 512, or cubes of k shards): 384 x 640^2, 640^3, 1152^3 (halo 64);
           --feather ink: k shards + 2 halo, k = 1..3 (576, 1088, 1600 at halo 32) */
        int cz[4], cw[4], nc = 0; const int ha = fea ? halo : 64;
        if (fea) { for (int k = 1; k <= 3; k++) { cz[nc] = cw[nc] = k * shard + 2 * ha; nc++; } }
        else { static const int fz[3] = {384, 640, 1152}, fx[3] = {640, 640, 1152}; for (int k = 0; k < 3; k++) { cz[nc] = fz[k]; cw[nc] = fx[k]; nc++; } }
        const int dir_ok = !getenv("UFSM_SLAB_DIRECT") || ufsm_env_on("UFSM_SLAB_DIRECT");
        const size_t fr = getenv("UFSM_PRED_WINDOW_FREE") ? (size_t)(atof(getenv("UFSM_PRED_WINDOW_FREE")) * 1e9) : nn_mem_free();   /* (planning: a GB figure, with UFSM_PRED_WINDOW_DRY=1) */
        int bk = 0, bd = dir_ok; double bc = 1e30;
        for (int k = 0; k < (wprod ? 1 : nc); k++) for (int d = dir_ok; d >= 0; d--) {
            const int iz = cz[k] - 2 * ha, ix = cw[k] - 2 * ha;
            if (iz <= 0 || ix <= 0) continue;
            const double cost = (double)cz[k] * cw[k] * cw[k] / ((double)iz * ix * ix) * (d ? 1.0 : fea ? 1.22 : 1.25);
            unet_set_slab_direct(d);
            const size_t w3 = (size_t)cz[k] * cw[k] * cw[k], e3 = fea ? (size_t)(ix + fea) * (ix + fea) * (ix + fea) : 0,
                         acc = ffz ? (size_t)iz * ix * ix / ((size_t)ffh * ffh * ffh) * 64 : 0,
                         need = unet_infer_bytes(u, (shape5){1, 4, cz[k], cw[k], cw[k]}) + w3 / 8 + e3 + acc + (tta > 1 ? 3 * w3 : 0) + (size_t)shard * shard * shard + ((size_t)3 << 29);   /* (--tta: the saved CT + the probability accumulator) */
            if ((need <= fr || (k == 0 && d == 0)) && cost < bc - 1e-9) { bc = cost; bk = k; bd = d; }
        }
        unet_set_slab_direct(bd ? 2 : 0);
        Wz = cz[bk]; W = cw[bk]; halo = hz = ha; slab = Wz != W;
        fprintf(stderr, "predict: --window auto%s: %d x %d^2, halo %d, %s slabs (%.1f GB free; %.2f computed voxels per interior voxel)\n", wprod ? " (prod)" : "",
                Wz, W, halo, bd ? "direct" : "copy", fr / 1e9, bc);
        if (ufsm_env_on("UFSM_PRED_WINDOW_DRY")) return 0;
    }
    if (split && !win_given && !wauto) { W = shard + 128; halo = 64; fprintf(stderr, "predict: window %d, halo %d (split default; --window / --halo set them; window - 2 halo = k * shard computes one window per k^3 shards)\n", W, halo); }
    if (split && W % (2 * H0)) { fprintf(stderr, "--split z: the window must be a multiple of %d\n", 2 * H0); return 2; }
    if (!split && !win_given && !wauto && !sheet_task && bn[0] >= shard && bn[1] >= shard && bn[2] >= shard) {   /* default window: one per k^3 shards (k shards + 2 halo,
                                          the largest k <= 4 the box and this GPU hold: in-place slab inference makes k = 2 fit 16 GB with the 16-wide net) or one per shard,
                                          halo 16 or the least multiple of 2^(levels-1) / 2 above it; else 288 / 8. Desk widths, 512 shards, laptop: 16% less GPU time per
                                          shard than 8 windows of 288, peak 9.4 GB */
        const int div = 1 << (cfg.nlev - 1), md = hdiv * div;
        const int ctin = !half && !ufsm_env_on("UFSM_PRED_HOSTPATH");   /* the device path puts the CT window in the level-0 skip buffer (unet_input_scratch2) */
        int hk = 16; while ((2 * hk) % md && hk < shard / 4) hk += 8;
        /* a latent-grid net writing --channel rv from k >= 2 shards a window holds no full-resolution tensor but the CT bytes (nn_lat_pred_input,
           nn_lat_place_rv): no 14 bytes a voxel in the estimate. k stays <= 4: with one 128^3 chunk per shard the run is bound by the shard
           writers and the CT reader, and windows of 8^3 shards skip less air (box 1024 x 3072^2 of PHerc1667: 4608 shards compressed against 3584) */
        const float *lb_, *lg_; float ls_;   /* (not --latents or the guided head: those hold the full-resolution input) */
        const int lat_lean = !cfg.fstem && rv2c < 0 && half && hdiv == 4 && hmode == 1 && (cfg.cin == 4 || (half == 3 && unet_lat_basis(u, &lb_, &lg_, &ls_) && nn_get_f16())) && rvfuse && !ffd && tta <= 1 && !fea && !auxz && nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8")
                             && (!getenv("UFSM_PRED_LAT_INPUT") || ufsm_env_on("UFSM_PRED_LAT_INPUT")) && (!getenv("UFSM_PRED_LAT_DIRECT") || ufsm_env_on("UFSM_PRED_LAT_DIRECT"));
        int bestk = -1; double bestv = 0;
        for (int k = 4; k >= 1; k--) {
            const int Wc = k * shard + 2 * hk;
            if (Wc % md || (k > 1 && (bn[0] < (int64_t)k * shard || bn[1] < (int64_t)k * shard || bn[2] < (int64_t)k * shard))) continue;
            const size_t wc3 = (size_t)Wc * Wc * Wc, need = (half ? unet_infer_bytes(u, (shape5){1, cfg.cin, Wc / hdiv, Wc / hdiv, Wc / hdiv}) + ((lat_lean && k > 1) || lhm ? 0 : 14) * wc3 + (size_t)nph * shard * shard * shard : unet_infer_bytes(u, (shape5){1, 4, Wc, Wc, Wc}))
                                                     + (ctin ? 0 : wc3) + wc3 / 8 + (size_t)shard * shard * shard + ((size_t)1 << 30);   /* (+ the CT window unless it lives in the network's level-0 buffer, its mask bits) */
            if (need > nn_mem_free()) continue;
            if (!lhm) { W = Wc; halo = hk; break; }
            /* the multi-head path: the k of the fewest computed voxels over the box (whole windows of k^3 shards; ties: the larger k), not the
               largest k that fits: 2048 x 3072^2 at k 3 computed 1.6x the voxels of k 2 (z: two 1536-voxel windows over 2048) */
            double cv = (double)wc3; for (int d = 0; d < 3; d++) cv *= (double)((bn[d] + (int64_t)k * shard - 1) / ((int64_t)k * shard));
            if (bestk < 0 || cv < bestv * 0.98) { bestk = k; bestv = cv; W = Wc; halo = hk; }
        }
        if (lhm && bestk > 0) fprintf(stderr, "predict: heads: k = %d (%.1f Gvox computed for the box's %.1f)\n", bestk, bestv / 1e9, (double)bn[0] * bn[1] * bn[2] / 1e9);
        if (W % md) W = (W + md - 1) / md * md;   /* (the 288 fallback, deeper nets: 320 / 8 with 7 levels) */
        fprintf(stderr, "predict: window %d, halo %d (default: one window per k^3 shards, k <= 4 the box and the GPU hold, else per shard, else 288 / 8; 288 / 16 with --fit-zarr; --window / --halo set them)\n", W, halo);
        if (box) for (int d = 0; d < 3; d++) if (boxv[d + 3] > INT64_MAX - W || boxv[d] > INT64_MAX - W - boxv[d + 3]) { fprintf(stderr, "prediction box exceeds the coordinate range\n"); return 2; }
    }
    if (ffz && !win_given && !wauto && !split && W == 288 && halo == 8) halo = 16;   /* --fit-zarr fallback: a z stride (256) dividing the shard */
    if (!win_given && !wauto && W % (hdiv << (cfg.nlev - 1))) W = (W + (hdiv << (cfg.nlev - 1)) - 1) / (hdiv << (cfg.nlev - 1)) * (hdiv << (cfg.nlev - 1));   /* (288 with a deeper net) */
    if (degrade && (slab || split || half || sheet_task || sweep_out || ufsm_env_on("UFSM_PRED_HOSTPATH") || W > 1024)) {
        fprintf(stderr, "predict: --aug-*: cube windows up to 1024 on one GPU's plain path\n"); return 2;
    }
    if (!slab) { Wz = W; hz = halo; }
    else if ((!win_given && !wauto) || Wz > W || Wz % (1 << (cfg.nlev - 1)) || hz >= Wz / 2 || split || half || sheet_task || auxz || sweep_out || ufsm_env_on("UFSM_PRED_HOSTPATH")) {
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
    {   /* the writer threads: --threads over the shard's chunks; shards of fewer chunks than threads several at a time (UFSM_PRED_WRITERS) */
        const int nc = (shard / 128) * (shard / 128) * (shard / 128) > 0 ? (shard / 128) * (shard / 128) * (shard / 128) : 1, nt = nthreads > 0 ? nthreads : (int)ufsm_ncpu();
        WR.nw = getenv("UFSM_PRED_WRITERS") ? atoi(getenv("UFSM_PRED_WRITERS")) : nt / nc;
        if (WR.nw < 1) WR.nw = 1;
        if (WR.nw > WRITERS_MAX) WR.nw = WRITERS_MAX;
        WR.wt = nt / WR.nw > 0 ? nt / WR.nw : 1;
        for (int i = 0; i < WR.nw; i++) pthread_create(&WR.wth[i], nullptr, writer_main, &WR);
    }
    int sb = 0; uint8_t *sbuf = WR.buf[0];
    /* the heads' pyramids on the GPU (pyacc_t) unless --resume (shards from an earlier run are not in memory) or UFSM_PRED_GPU_PYR=0 */
    /* the heads' chunks encoded from levels the GPU computes (vcgpu.h: volcomp_encode's streams; the CPU tokenises and entropy-codes),
       lossy q only (UFSM_PRED_GPU_ENC=0: the CPU encodes from the bytes) */
    const int genc = lhm && gpath && q > 0 && !(shard % 128) && (!getenv("UFSM_PRED_GPU_ENC") || ufsm_env_on("UFSM_PRED_GPU_ENC"));
    float *genc_rs = nullptr; uint16_t *genc_sc = nullptr; int genc_fma = 0;
    if (genc) { float rs[4096]; uint16_t sc[4096]; genc_fma = vcg_tables(q, rs, sc); genc_rs = nn_malloc(sizeof rs); genc_sc = nn_malloc(sizeof sc);
                nn_h2d(genc_rs, rs, sizeof rs); nn_h2d(genc_sc, sc, sizeof sc); }
    const int gpyr = lhm && nlev > 1 && !resume && !fullg && gpath && (!getenv("UFSM_PRED_GPU_PYR") || ufsm_env_on("UFSM_PRED_GPU_PYR")); size_t hab = WR.bytes;
    if (lhm) {   /* the heads' stores (--heads: <out>/<name>; one head: the main store), writers and device shards; the main writer writes nothing */
        if (hspec && resume) { fprintf(stderr, "predict --heads: no --resume (each head's store has its own shards)\n"); return 2; }
        WR.skip = 1;
        for (int k = 0; k < nph; k++) {
            phead_t *h = &PH[k];
            snprintf(h->attrs, sizeof h->attrs, "{\"ufsm\":{\"content\":\"%s (uint8 = round(p*255))\",\"head\":\"%s\",\"channel\":%d,\"checkpoint\":\"%s\",\"step\":%d,\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d,\"ema\":%d%s%s}}",
                     h->content, h->name, h->c0, ckpt, step, root, ak, (long long)(fullg ? 0 : bo[0]), (long long)(fullg ? 0 : bo[1]), (long long)(fullg ? 0 : bo[2]), W, halo, use_ema, grid_attrs, input_attrs);
            if (hspec) { char d[1600]; snprintf(h->dir, sizeof h->dir, "%s/%s", out, h->name); snprintf(d, sizeof d, "%s/%s", h->dir, lv);
                         if (!(h->w = z3w_create(d, ash, shard, q, 0, h->attrs))) { fprintf(stderr, "%s\n", z3w_error()); return 1; } }
            else { h->w = w; snprintf(h->dir, sizeof h->dir, "%s", out); }
            h->pya = nullptr;
            if (gpyr) { h->pya = malloc(sizeof(pyacc_t)); if (!h->pya || pyacc_init(h->pya, h->dir, um_l, ash, shard, nlev, q, h->attrs)) { fprintf(stderr, "predict: head %s: the GPU pyramid's stores\n", h->name); return 1; } }
            if (h->pya) { const int nt = (nthreads > 0 ? nthreads : (int)ufsm_ncpu()) / nph; h->pya->nthreads = nt > 1 ? nt : 1; }   /* (the heads' writers run at once) */
            size_t ab = h->pya ? h->pya->poff[h->pya->nlev] : WR.bytes, db = ab; const size_t voff = (ab + 255) / 256 * 256;
            vcg_layout VL; const size_t vcap = (size_t)shard * shard * shard / 8;   /* (levels: q 8 heads ~2 % of the coefficients; above 12.5 %: the bytes) */
            if (genc) { vcg_layout_of(&VL, shard, vcap); ab = voff + VL.bytes; db = voff + VL.o_lv; }
            hab = ab;
            h->wr = malloc(sizeof(writer_t)); h->dsh2[0] = nn_malloc(db); h->dsh2[1] = nn_malloc(db);
            if (!h->wr || !h->dsh2[0] || !h->dsh2[1] || phead_writer(h->wr, h->w, WR.bytes, ab, nthreads, 2, gpu, WR.async)) { fprintf(stderr, "predict: out of memory for head %s\n", h->name); return 1; }
            h->vrs = nullptr;
            if (genc) { h->wr->venc = 1; h->wr->vn = shard; h->wr->voff = voff; h->wr->vcap = vcap; h->wr->vq = q; h->vrs = genc_rs; h->vsc = genc_sc; h->vfma = genc_fma; h->vhdr = VL.o_lv; }
            for (int d = 0; d < 3; d++) h->wr->off[d] = soff[d];
            if (h->pya) { h->wr->post = pyacc_post; h->wr->pctx = h->pya; }
        }
        if (!hspec) { PH[0].wr->donefd = WR.donefd; WR.donefd = -1; }   /* (--resume: the head's writer logs the shards it wrote) */
    }
    /* the window goes up as uint8; the input channels are built on the device in the network's storage type */
    const int h16 = nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8");
    uint8_t *pu = gpath ? nullptr : malloc(w3), *pud = gpath ? nullptr : nn_malloc(w3);
    float *dyo = malloc(2 * (size_t)W * sizeof(float)), *dxo = dyo + W, *dyd = nn_malloc(2 * (size_t)W * sizeof(float)), *dxd = dyd + W;
    const int Dl = split ? W / 2 + H0 : Wz;   /* planes per GPU */
    shape5 xs = {1, !half && (cfg.cin == 5 || cfg.cin == 7) ? cfg.cin : 4, Dl, W, W};   /* (7: a train --cond net, its condition inputs zero here) */
    const shape5 xn = half ? (shape5){1, cfg.cin, W / hdiv, W / hdiv, W / hdiv} : xs;   /* the network's input */
    int latin = 0; void *x2 = nullptr, *lrt = nullptr; float *lgf = nullptr;   /* half: the network's input, the intermediate grid (div 4), the full-resolution logits */
    /* the 16-bit input in the network's shared a1 scratch when it has one (free until the stem conv, after the input conversion) */
    unet_build(u, xn, 0);
    const int ct_stem = unet_ct_stem(u);   /* unet_forward_ct: no 16-bit input */
    int gdir = 0, fzs = 0;   /* the guided head's direct rv bytes (nn_lat_place_rvg); fit fields from a latent-grid net in z-chunk slabs */
    void *fzA = nullptr, *fzB = nullptr, *fzX = nullptr; int fzP = 0;
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
        /* a latent-grid net's input goes from the CT bytes straight to the cell grid (nn_lat_pred_input): no full-resolution 4-channel
           window (8 bytes a voxel) or half-resolution intermediate (UFSM_PRED_LAT_INPUT=0: nn_pred_input_d + nn_lowres_input) */
        /* --fit-zarr from a latent-grid net (--no-out 1, no --fit-ct: its ray reads the whole window): the heads upsampled (and guided) only over
           the planes of each z chunk of field cells, not [cout][W^3] fp16 (10 GB at W 640); UFSM_PRED_FIT_SLAB=0: the whole window */
        fzs = ffd && ffzm && half && hdiv == 4 && hmode == 1 && h16 && gpath && !ffct && tta <= 1 && !fea && !sheet_task && !slab && WR.skip && !(W % 4) && (!getenv("UFSM_PRED_FIT_SLAB") || ufsm_env_on("UFSM_PRED_FIT_SLAB"));
        gdir = !cfg.fstem && rv2c < 0 && (cfg.lat_guide == 5 || cfg.lat_guide == 64) && half && hdiv == 4 && hmode == 1 && rvfuse && gpath && !fea && !ffd && tta <= 1 && !sheet_task && !slab && (!getenv("UFSM_PRED_LAT_DIRECT") || ufsm_env_on("UFSM_PRED_LAT_DIRECT"));
        if (((gdir && cfg.lat_guide == 5) || (fzs && (cfg.lat_guide == 8 || cfg.fstem)) || lhm) && ct_shared && !ctcp && !(ctcp = nn_malloc(w3))) { fprintf(stderr, "predict: no device memory for the guided head's CT copy\n"); return 1; }   /* (the forward overwrites ctd) */
        { const float *b_, *g_; float s_; const int lc_ = half == 3 ? unet_lat_basis(u, &b_, &g_, &s_) : 0;   /* (--latents and the guided head's slab path read the full-resolution input; a
                                                                                                                 --lat-basis checkpoint: nn_lat_pred_input_c, fp16 storage) */
          latin = half && hdiv == 4 && hmode == 1 && xs.c == 4 && (lc_ ? nn_get_f16() != 0 : cfg.cin == 4) && h16 && !auxz && !sheet_task && (!cfg.fstem || fzs || (lhm && !ffd)) && (!cfg.lat_guide || cfg.lat_guide == 64 || gdir || fzs)   /* (the sub-cell head reads no CT; the fine stem does: from the CT bytes in the multi-head and fit-slab paths) */
                  && (!getenv("UFSM_PRED_LAT_INPUT") || ufsm_env_on("UFSM_PRED_LAT_INPUT")); }
        if (!xd && !ct_stem && !latin) xd = nn_malloc((size_t)xs.c * w3 * (h16 ? 2 : 4));
        if (auxz) { auxd = nn_malloc(w3); auxh = malloc(w3); }
        if (half) { x2 = nn_malloc(shape_numel(xn) * (h16 ? 2 : 4));   /* (lgf, and lrt with the direct latent input: at their first use; the direct latent rv path has none) */
            if (hdiv > 2 && !latin) lrt = nn_malloc(w3 / 8 * 16); }
        if (lat_store && gpath) lz_d = nn_malloc(w3 / 64 * (lz_c + 1));   /* (a voxel checkpoint too: its CT window rebuilt from the cells) */
        /* a latvol CT array (src/latvol.h): the window's chunk streams go up as stored and the device decodes them into the cells, from
           which the CT window is rebuilt (every reader of the CT bytes) and, when the checkpoint's transform is the store's, the trunk's
           input is taken (no CT -> latent conversion, no host decode). Host-side CT readers (--aug-*, --aux, --split z, --tta, the host
           path) and windows off the 4-voxel cell grid read the rebuilt CT through z3_read instead. UFSM_PRED_LATVOL=0: always z3_read */
        if (!lat_store && lvc && gpath && !degrade && !auxz && !split && tta <= 1 && W % 4 == 0 && Wz % 4 == 0 && (!getenv("UFSM_PRED_LATVOL") || ufsm_env_on("UFSM_PRED_LATVOL"))) {
            lvm = 1; lv_stride = lvc->C + 1; lz_d = nn_malloc(w3 / 64 * lv_stride);
        }
        if (ct_stem) fprintf(stderr, "predict: in-place slab inference from the CT window (%.1f B per window voxel)\n", (double)unet_activation_bytes(u) / w3);
        if (tta > 1 && (!ct_stem || !ct_shared || degrade || auxz || cfg.cin != 4)) { fprintf(stderr, "predict: --tta needs the CT-stem device path (no --aug-*, --aux, extra inputs)\n"); return 2; }
    }
    int64_t ns[3];
    for (int d = 0; d < 3; d++) ns[d] = (bn[d] + shard - 1) / shard;
    /* --resume / --occupancy: the box's shards not to compute (1 = already written, 2 = air: the outputs are masked to CT > 0, so a shard
       whose own region is zero on the coarse level, one coarse voxel of margin, predicts nothing) */
    uint8_t *sskip = nullptr; long nresumed = 0, nocc = 0;
    uint8_t *occb = nullptr; int64_t occo[3] = {0, 0, 0}, occn[3] = {0, 0, 0}; int occf = 1;   /* --occupancy: the coarse level over the box (tile tests below) */
    if (resume || occl >= 0 || lat_store) {
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
        if (occl >= 0 || lat_store) {
            /* --latents: the occupancy of the store's chunks (a chunk of 32^3 cells is 128^3 voxels; absent / all-zero chunks are air, known
               from the shard indices: no CT and no cell data read) on a grid of f = 32 voxels, each chunk's material marking every grid voxel
               it touches (a superset: the store's origin need not be a multiple of 32); else the coarse CT level --occupancy L */
            z3 *oc = lat_store ? nullptr : pyramid_open_level(s, key, occl, um, cache);
            if (!lat_store && !oc) { fprintf(stderr, "predict: --occupancy: cannot open CT level %d: %s\n", occl, z3_error()); return 1; }
            const int f = lat_store ? 32 : 1 << (occl - level);
            int64_t o0[3], n0[3];
            for (int d = 0; d < 3; d++) {
                int64_t lo = bo[d] / f - 1, hi = (bo[d] + bn[d] + f - 1) / f + 1;
                if (lo < 0) lo = 0;
                if (!lat_store && hi > z3_meta_of(oc)->shape[d]) hi = z3_meta_of(oc)->shape[d];
                o0[d] = lo; n0[d] = hi > lo ? hi - lo : 0;
            }
            uint8_t *ob = n0[0] && n0[1] && n0[2] ? (lat_store ? calloc((size_t)(n0[0] * n0[1] * n0[2]), 1) : malloc((size_t)(n0[0] * n0[1] * n0[2]))) : nullptr;
            if (lat_store && ob) {
                const int64_t cvx = 4 * (int64_t)lzr.chunk; long nch = 0, nmat = 0;
                int64_t c0[3], c1[3];
                for (int d = 0; d < 3; d++) { int64_t a = bo[d] - lz_org[d] - f; c0[d] = a < 0 ? 0 : a / cvx; c1[d] = (bo[d] + bn[d] - lz_org[d] + f + cvx - 1) / cvx; if (c1[d] * lzr.chunk > lzr.shape[d]) c1[d] = (lzr.shape[d] + lzr.chunk - 1) / lzr.chunk; }
                for (int64_t cz = c0[0]; cz < c1[0]; cz++) for (int64_t cy = c0[1]; cy < c1[1]; cy++) for (int64_t cx = c0[2]; cx < c1[2]; cx++) {
                    nch++;
                    if (zr_chunk_empty(&lzr, cz * lzr.chunk, cy * lzr.chunk, cx * lzr.chunk) != 0) continue;
                    nmat++;
                    const int64_t cc[3] = {cz, cy, cx}; int64_t a[3], b[3];
                    for (int d = 0; d < 3; d++) {
                        const int64_t g0 = lz_org[d] + cc[d] * cvx, g1 = g0 + cvx;
                        a[d] = g0 / f - o0[d]; b[d] = (g1 - 1) / f + 1 - o0[d];
                        if (a[d] < 0) a[d] = 0;
                        if (b[d] > n0[d]) b[d] = n0[d];
                    }
                    for (int64_t z = a[0]; z < b[0]; z++) for (int64_t y = a[1]; y < b[1]; y++) memset(ob + ((size_t)z * n0[1] + y) * n0[2] + a[2], 1, b[2] > a[2] ? (size_t)(b[2] - a[2]) : 0);
                }
                fprintf(stderr, "predict: --latents occupancy: %ld of %ld store chunks near the box hold material\n", nmat, nch);
            }
            if (n0[0] && n0[1] && n0[2] && (!ob || (!lat_store && z3_read(oc, o0, n0, ob, nthreads)))) { fprintf(stderr, "predict: --occupancy: cannot read CT level %d: %s\n", occl, z3_error()); return 1; }
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
            if (oc) z3_close(oc);
        }
        fprintf(stderr, "predict: %lld shards in the box: %ld already written (--resume), %ld air on %s %d (--occupancy), %lld to compute\n",
                (long long)(ns[0] * ns[1] * ns[2]), nresumed, nocc, lat_store ? "the latent store, grid" : "CT level", lat_store ? 32 : occl, (long long)(ns[0] * ns[1] * ns[2]) - nresumed - nocc);
    }
    /* interior = k whole shards (k >= 2, the tile grid on the shard grid): one window per k^3 shards, each shard placed from it and
       written (INFERENCE_MEMORY_PLAN.md, memory into speed); otherwise one or more windows per shard, clipped to it */
    int kss = !slab && !sheet_task && gpath && stride % shard == 0 && stride / shard >= 2 ? stride / shard : 1;
    for (int d = 0; d < 3 && kss > 1; d++) { int64_t ph = grid ? (bo[d] - grid_origin[d]) % stride : 0; if (ph < 0) ph += stride; if (ph) kss = 1; }
    if (fea) {   /* --feather: the window grid on the shard grid, interior k shards (k >= 1), one GPU path */
        int bad = fea < 0 || fea > halo || fea > hz || fea > 1024 || slab || sheet_task || !gpath || split || (ffd && (!ffzm || tta > 1)) || half || stride % shard || sweep_out;
        for (int d = 0; d < 3 && !bad; d++) { int64_t ph = grid ? (bo[d] - grid_origin[d]) % stride : 0; if (ph < 0) ph += stride; if (ph) bad = 1; }
        if (bad) { fprintf(stderr, "predict: --feather B needs 0 < B <= halo (<= 1024), window - 2 halo = k * shard with the box on that grid, the plain device path "
                               "(no --split, --fit-fields, --fit-zarr with --tta, --window-z, half-resolution or winding outputs)\n"); return 2; }
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
              if (ffd && ffzm && half && hdiv == 4 && WR.skip && !ffct && tta <= 1) {   /* a latent-grid net's z-chunk slabs (fzs): <= ~768 MB of fp16 head planes a chunk */
                  const size_t pls = ((size_t)768 << 20) / ((size_t)(cfg.lat_guide == 8 ? 19 : cfg.lat_guide == 64 ? 11 : cfg.cout) * W * W * 2), czc = pls / (size_t)ffh;
                  if ((size_t)ffzc > (czc ? czc : 1)) ffzc = (int)(czc ? czc : 1);
              }
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
        if (lhm) {   /* each head's ring: 2 k^3 buffers within UFSM_PRED_HBUF_GB (8) of pinned memory over all heads */
            const double gb = getenv("UFSM_PRED_HBUF_GB") ? atof(getenv("UFSM_PRED_HBUF_GB")) : 8.0; int hb = sup ? 2 * kss * kss * kss : 2;
            const int cap = (int)(gb * 1e9 / ((double)nph * hab)); if (hb > cap) hb = cap; if (hb < 2) hb = 2; if (hb > WBUF_MAX) hb = WBUF_MAX;
            for (int j = 0; j < nph; j++) { writer_t *Wh = PH[j].wr; int n2 = Wh->nb;
                for (int k = n2; k < hb; k++) { if (!(Wh->buf[k] = nn_host_alloc_try(hab))) break; n2 = k + 1; }
                pthread_mutex_lock(&Wh->mu); Wh->nb = n2; pthread_mutex_unlock(&Wh->mu); }
            fprintf(stderr, "predict: %d head stores, %d shard buffers each (%.1f GB pinned)%s%s\n", nph, PH[0].wr->nb, nph * PH[0].wr->nb * (double)hab / 1e9, gpyr ? ", pyramids pooled on the GPU" : "", genc ? ", levels for the encoder from the GPU" : "");
        }
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
    if (lat_store) {   /* every window on the store's cell grid */
        if (W % 4 || Wz % 4) { fprintf(stderr, "predict: --latents: --window (and --window-z) must be multiples of 4\n"); return 2; }
        for (long i = 0; i < rd.ntiles; i++) {
            const int64_t o[3] = {bo[0] + rd.tiles[i].tz, bo[1] + rd.tiles[i].ty, bo[2] + rd.tiles[i].tx};
            for (int d = 0; d < 3; d++) if ((o[d] - lz_org[d]) % 4) {
                fprintf(stderr, "predict: --latents: window origin %lld,%lld,%lld is off the store's 4-voxel cell grid (origin %lld,%lld,%lld): use a box origin, "
                                "--grid-origin, halo and stride that are multiples of 4 from it\n", (long long)o[0], (long long)o[1], (long long)o[2], lz_org[0], lz_org[1], lz_org[2]);
                return 2;
            }
        }
        rd.lzt = nthreads < 1 ? 1 : nthreads > 32 ? 32 : nthreads; rd.lz = calloc((size_t)rd.lzt, sizeof(zr)); rd.lzo = lz_org; rd.lzc = lz_c;
        for (int t = 0; t < rd.lzt; t++) if (zr_open(&rd.lz[t], lat_store)) { fprintf(stderr, "predict: --latents: cannot open %s\n", lat_store); return 1; }
        const float *lg_; float ls_; const int lc_ = unet_lat_basis(u, &lz_bas, &lg_, &ls_);
        if (!lc_ && !half && gpath && lz_d) {   /* a voxel checkpoint (no --lat-basis): fed the CT rebuilt from the store's cells with the store's own transform
                                                    (the codec test: a CT net on the decoded CT against the real one) */
            char bp[1200]; snprintf(bp, sizeof bp, "%s/basis.lcb", lat_store); lat_basis *lb = malloc(sizeof *lb);
            if (!lb || lat_basis_load(bp, lb)) { fprintf(stderr, "predict: --latents: cannot read %s\n", bp); return 1; }
            float *db = nn_malloc((size_t)lz_c * 64 * 4); nn_h2d(db, lb->b, (size_t)lz_c * 64 * 4); free(lb); lz_bas = db;
            fprintf(stderr, "predict: --latents with a voxel checkpoint: its CT windows rebuilt from the %d stored components\n", lz_c);
        } else
        if (!gpath || lc_ != lz_c || !lz_d) { fprintf(stderr, "predict: --latents needs the device path and a --lat-basis checkpoint with the store's %d cell channels (it has %d)\n", lz_c, lc_); return 2; }
        {   /* the store's transform must be the checkpoint's: the rebuilt CT and the channels both assume it */
            char bp[1200]; snprintf(bp, sizeof bp, "%s/basis.lcb", lat_store); lat_basis *lb = malloc(sizeof *lb); float *hb = malloc((size_t)lz_c * 64 * 4); double dm = 0;
            if (!lb || !hb || lat_basis_load(bp, lb)) { fprintf(stderr, "predict: --latents: cannot read %s\n", bp); return 1; }
            nn_d2h(hb, lz_bas, (size_t)lz_c * 64 * 4);
            for (int c = 0; c < lz_c; c++) for (int k2 = 0; k2 < 64; k2++) dm = fmax(dm, fabs((double)hb[c * 64 + k2] - lb->b[c][k2]));
            free(lb); free(hb);
            if (dm > 1e-5) { fprintf(stderr, "predict: --latents: the store's transform differs from the checkpoint's (max |db| %.3g)\n", dm); return 2; }
        }
    }
    if (lvm) {
        for (long i = 0; i < rd.ntiles && lvm; i++) if ((bo[0] + rd.tiles[i].tz) % 4 || (bo[1] + rd.tiles[i].ty) % 4 || (bo[2] + rd.tiles[i].tx) % 4) lvm = 0;
        if (!lvm) fprintf(stderr, "predict: latvol CT: windows off the 4-voxel cell grid (box origin, halo, stride): the rebuilt CT is decoded on the host instead\n");
    }
    if (lvm) {
        lvL = nn_latvol_create(lvc->dt, lvc->ntab, lvc->map, &lvc->bi[0][0], lvc->C, lvc->step_raw);
        const float *lb_, *lg_; float ls_; const int lc_ = half == 3 ? unet_lat_basis(u, &lb_, &lg_, &ls_) : 0;
        if (lc_ && lc_ <= lvc->C) {   /* the trunk's channels straight from the indices when the store's first rows are the checkpoint's transform */
            float *hb = malloc((size_t)lc_ * 64 * 4); double dm = 0; nn_d2h(hb, lb_, (size_t)lc_ * 64 * 4);
            for (int c = 0; c < lc_; c++) for (int k2 = 0; k2 < 64; k2++) dm = fmax(dm, fabs((double)hb[c * 64 + k2] - lvc->basis[c][k2]));
            free(hb); lvq = dm <= 1e-5 && !(getenv("UFSM_PRED_LATVOL") && atoi(getenv("UFSM_PRED_LATVOL")) == 2);   /* (2: the check mode, every input from the device-rebuilt CT = UFSM_PRED_LATVOL=0's bytes) */
            if (!lvq) fprintf(stderr, "predict: latvol CT: the store's transform differs from the checkpoint's (max |db| %.3g): the trunk's input from the rebuilt CT\n", dm);
        }
        rd.lv = 1; rd.lvmax = (int)((Wz / 128 + 2) * (W / 128 + 2) * (W / 128 + 2));
        for (int k = 0; k < NSLOT; k++) rd.lvref[k] = malloc((size_t)rd.lvmax * sizeof(lv_chunk_ref));
        fprintf(stderr, "predict: latvol CT (C %d, step %.2f grey levels): cells decoded on the device, %s\n", lvc->C, lvc->step_raw,
                lvq ? "the trunk's input from the stored indices" : "every input from the rebuilt CT");
    }
    for (int k = 0; k < NSLOT; k++) rd.buf[k] = gpath ? nn_host_alloc(w3) : malloc(w3);
    if (devread && !lvm && gpath && !degrade && !lat_store && !split) {   /* UFSM_PRED_DEVREAD: device windows (else the host path) */
        const z3_meta *cm = z3_meta_of(ct); int ok = cm->chunk[0] == 128 && cm->chunk[1] == 128 && cm->chunk[2] == 128 && cm->esize == 1 && gpudec_enabled();
        for (int k = 0; k < NSLOT && ok; k++) ok = (rd.dbuf[k] = nn_malloc_try(w3)) != nullptr;
        /* UFSM_PRED_DEVREAD_GPU=G workers feed the GPU decoder (default 16), --threads more decode on the CPU (pod, 19 Gvox: all 40 on the GPU
           decoder read 10.1 s, the host reader 5.6: the batched GPU round trips leave the CPU idle) */
        const int nc = nthreads > 0 ? nthreads : (int)ufsm_ncpu(), ng = getenv("UFSM_PRED_DEVREAD_GPU") ? atoi(getenv("UFSM_PRED_DEVREAD_GPU")) : 16, nw = (ng > 0 ? ng : 0) + nc;
        rd.ngpuw = ng > 0 ? ng : 0;
        if (ok) { rd.ctw = calloc((size_t)nw, sizeof(void *)); for (int t = 0; t < nw && ok; t++) ok = (rd.ctw[t] = nn_ctw_new(gpu)) != nullptr; rd.nctw = nw; }
        for (int d = 0; d < 3; d++) rd.csh[d] = cm->shape[d];
        if (!ok) { for (int k = 0; k < NSLOT; k++) { if (rd.dbuf[k]) nn_free(rd.dbuf[k]); rd.dbuf[k] = nullptr; }
                   fprintf(stderr, "predict: UFSM_PRED_DEVREAD: no device windows (memory, chunking or no GPU decoder): the host path\n"); }
        else fprintf(stderr, "predict: the CT windows decoded into device memory (%d chunk workers on the GPU decoder, %d on the CPU, %.1f GB)\n", rd.ngpuw, nw - rd.ngpuw, NSLOT * (double)w3 / 1e9);
    }
    rd.nostats = gpath;
    atomic_store(&rd.filled, 0); atomic_store(&rd.consumed, 0); atomic_store(&rd.failed, 0);
    pthread_mutex_init(&rd.mu, nullptr); pthread_cond_init(&rd.cv, nullptr);
    pthread_t rth; pthread_create(&rth, nullptr, reader_main, &rd);
    static prefetch_t PF; pthread_t pfth; int pf_on = 0;
    {   const char *pe = getenv("UFSM_PRED_PREFETCH"); const int ahead = pe ? atoi(pe) : 2;
        if (ahead > 0 && cache && !strncmp(root, "http", 4) && !lat_store && rd.ntiles > 1) {
            PF = (prefetch_t){.rd = &rd, .ahead = ahead, .nt = getenv("UFSM_PRED_PREFETCH_THREADS") ? atoi(getenv("UFSM_PRED_PREFETCH_THREADS")) : 16};
            if (!pthread_create(&pfth, nullptr, prefetch_main, &PF)) pf_on = 1;
        } }
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
    static pyr_pipe PP; pyr_pipe *pp = nullptr;
    if (fea && gpath && sup && nlev > 1 && !WR.skip && nwork <= 1 && !split && !sheet_task && (!getenv("UFSM_PYR_PIPE") || ufsm_env_on("UFSM_PYR_PIPE"))) {
        const int64_t n1 = ((ash[0] + 1) / 2 + shard - 1) / shard;
        PP = (pyr_pipe){.out = out, .um = um_l, .shape0 = ash, .shard = shard, .nthreads = nthreads, .q = q, .attrs = attrs, .WR = &WR, .FW = &FW,
                        .nz1 = n1, .done = calloc((size_t)n1, 1)};
        PP.jq = malloc((size_t)(n1 + 1) * sizeof *PP.jq);
        pthread_mutex_init(&PP.mu, nullptr); pthread_cond_init(&PP.cv, nullptr);
        if (PP.done && PP.jq && !pthread_create(&PP.th, nullptr, pyr_main, &PP)) pp = &PP;
    }
    int64_t cur_shard = -1, so[3] = {0, 0, 0}, se[3] = {0, 0, 0}, sz = 0, sy = 0, sx = 0;
    int any = 0;
    const int pprof = ufsm_env_on("UFSM_PRED_PROF");
    double p_wait = 0, p_h2d = 0, p_gpu = 0, p_copy = 0, p_wr = 0, p_ffw = 0, pt, lv_up = 0;
    /* the next window's CT uploaded on the copy stream during this window's forward (into ctpre, w3 bytes; at the window's start one
       device copy into ctd): the synchronous upload left the GPU idle ~9 ms per 384 x 640^2 window. Not with --aug-* (degrade rewrites
       the host window first); UFSM_PRED_CTPRE=0 off. long pre_i: the window whose CT ctpre holds (or is receiving) */
    uint8_t *ctpre = nullptr; void *ev_pre = nullptr, *ev_preu = nullptr; long pre_i = -1;
    if (gpath && !degrade && !split && !lat_store && !lvm && !rd.dbuf[0] && (!getenv("UFSM_PRED_CTPRE") || ufsm_env_on("UFSM_PRED_CTPRE")) && (ctpre = nn_malloc_try(w3))) {
        ev_pre = nn_event_create(); ev_preu = nn_event_create();
    }
    if (split) {   /* both GPUs on every window */
        J.rd = &rd; J.WR = &WR; J.tiles = rd.tiles; J.ntiles = rd.ntiles;
        J.W = W; J.halo = halo; J.shard = shard; J.Dl = Dl; J.H0 = H0; J.kss = kss; J.h16 = h16; J.sign = (int)radial_sign;
        for (int d = 0; d < 3; d++) { J.ns[d] = ns[d]; J.nss[d] = nss[d]; J.bn[d] = bn[d]; J.bo[d] = bo[d]; }
        J.ax = &ax; J.scale = scale; J.t0 = t0; J.pcut = pcut; J.pcond = pap_cond; J.norm_mean = norm_mean; J.norm_sd = norm_sd;
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
        if (pp) pyr_advance(pp, t->shard / (nss[1] * nss[2]) * kss + soff[0], fe_n);   /* (feather windows: z-y-x, one shard each) */
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
                if (any && lhm) { for (int j = 0; j < nph; j++) if (phead_submit(&PH[j], phd, sz, sy, sx, sz)) return 1; phd ^= 1; }
                if (any) { if (gpath && !WR.skip) writer_d2h(&WR, sb, dsh); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb = writer_next(&WR, sb); sbuf = WR.buf[sb]; }
                else if (WR.donefd >= 0) dprintf(WR.donefd, "%lld %lld %lld\n", (long long)(sz + soff[0]), (long long)(sy + soff[1]), (long long)(sx + soff[2]));   /* all air: done */
                p_wr += now() - pt;
                fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   ", (long long)(cur_shard + 1), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
                if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
            }
            cur_shard = t->shard; sz = cur_shard / (ns[1] * ns[2]); sy = (cur_shard / ns[2]) % ns[1]; sx = cur_shard % ns[2];
            so[0] = sz * shard; so[1] = sy * shard; so[2] = sx * shard;
            for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
            if (gpath) { if (!WR.skip) nn_zero(dsh, WR.bytes); for (int j = 0; j < nph; j++) phead_fresh(&PH[j], phd); } else memset(sbuf, 0, (size_t)shard * shard * shard);
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
                nn_ptick("wait reader + host");
                pt = now();
                if (lat_store) {   /* --latents: the window's cells up, its CT rebuilt from them on the device (air windows: nothing) */
                    if (rd.nz[k]) { nn_h2d(lz_d, ctu, w3 / 64 * (lz_c + 1)); nn_lat_synth_ct(lz_d, lz_c, lz_bas, (float)lz_sraw, Wz, W, ctd); nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
                                  sq += lz_drop * (double)nz; }   /* (the dropped axes' energy in the material voxels: the CT window's variance) */
                    else nz = 0;
                } else if (lvm) {   /* the packed streams up (into the CT window's own buffer: dead once decoded), the cells, the CT rebuilt from them */
                    if (rd.nz[k]) {
                        nn_h2d(ctd, ctu, rd.lvbytes[k]);
                        if (nn_latvol_decode(lvL, ctd, rd.lvref[k], rd.lvn[k], Wz / 4, W / 4, W / 4, lv_stride, lz_d)) { fprintf(stderr, "\npredict: latvol: a corrupt chunk stream in window %lld,%lld,%lld\n", (long long)o[0], (long long)o[1], (long long)o[2]); return 1; }
                        nn_latvol_synth(lvL, lz_d, lv_stride, Wz / 4, W / 4, W / 4, ctd); nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
                        sq += (double)lvc->drop_ms * (double)nz; lv_up += (double)rd.lvbytes[k];
                    } else nz = 0;
                } else {
                if (pre_i == i) {   /* prefetched: the copy stream's upload, then one device copy (the prefetch buffer free again after it) */
                    nn_stream_wait(0, ev_pre); nn_d2d_async(ctd, ctpre, w3); nn_event_record(ev_preu, 0);
                } else if (rd.dbuf[0]) nn_d2d(ctd, rd.dbuf[k], w3); else nn_h2d(ctd, ctu, w3);
                nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
                if (lvc) sq += (double)lvc->drop_ms * (double)nz;   /* (a latvol CT decoded by z3_read: the rebuilt bytes' variance lacks the dropped axes, as below) */
                }
                pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu);
                p_h2d += now() - pt; nn_ptick("upload + CT stats");
            }
            if (nz == 0) { nskip++; pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu); continue; }
            double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
            if (norm_sd > 0) { mean = norm_mean; sd = norm_sd; }   /* --norm: one fixed z-score for every window */
            { static int dbg = -1; if (dbg < 0) dbg = ufsm_env_on("UFSM_LAT_DEBUG") ? 4 : 0; if (dbg > 0) { dbg--; fprintf(stderr, "\npredict: window %lld,%lld,%lld mean %.3f sd %.3f (%zu nonzero)%s\n", (long long)o[0], (long long)o[1], (long long)o[2], mean, sd, nz, lat_store ? " --latents" : ""); } }
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
            if (!ct_stem && !latin) nn_pred_input_d(ctd, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, xd, h16);
            nn_ptick("input build");
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
                                .axis = ax.n > 0 ? (int)radial_sign : 0, .xs = xs, .h16 = h16, .D = Wz, .W = W, .nch = ffd ? cfg.cout : 1, .ch_normal = ffd ? ff_nrm : -1};   /* (without fit fields the head computes only the written, scalar channel) */
            if (ct_shared && ctmi != ctm) nn_pred_ctmask_t(ctd, w3, ctmi, iso);
            if (ctcp && ct_shared) nn_d2d_async(ctcp, ctd, w3);   /* --fit-ct reads the CT after the forward */
            if (sheet_task) {
                for (int z=0;z<W;z++) { double a[4]; sheet_parameters(sheet,o[0]+z,a); sheet_rows[4*z]=(float)(o[1]-a[0]); sheet_rows[4*z+1]=(float)(o[2]-a[1]); sheet_rows[4*z+2]=(float)(1/a[2]); sheet_rows[4*z+3]=(float)a[3]; }
                nn_sheet_input(xd,W,sheet_rows,(float)sheet->center,(float)sheet->scale,h16);
            }
            pred_axis(u, ax.n > 0, Wz, dyo, dxo);
            pred_mem_in(u, o, W);
            if (pred_view_set(u, o, Wz, W)) return 1;
            if (g_pv && tta > 1) nn_d2d(g_pvsave, g_pvd, (size_t)g_pvP * g_pvP * g_pvP);
            const float *lg;
            /* --channel rv on a latent-grid net, one window per k^3 shards: the cell grid's two side logits go straight into each shard's
               bytes (nn_lat_place_rv; the guided head: nn_lat_place_rvg, its CT terms from the bytes); no full-resolution logits
               (UFSM_PRED_LAT_DIRECT=0: the fp32 / guided fp16 volume, nn_rv_fuse, nn_pred_place). Not for the fit fields (every head at full resolution) */
            static int latdir = -1; if (latdir < 0) latdir = !getenv("UFSM_PRED_LAT_DIRECT") || ufsm_env_on("UFSM_PRED_LAT_DIRECT");
            const int ldir = !cfg.fstem && latdir && rv2c < 0 && half && hdiv == 4 && hmode == 1 && rvfuse && (!cfg.lat_guide || gdir) && gpath && sup && !fea && !ffd && tta <= 1 && !sheet_task && Wz == W;
            const float *l2d = nullptr;
            if (half) {   /* the network on blocks of the window: its input from the 4-channel window, its logits back to full resolution */
                const float *lbas, *lgain; float lstep; const int latc = half == 3 ? unet_lat_basis(u, &lbas, &lgain, &lstep) : 0;
                if (latin && cfg.lat_guide && cfg.lat_guide != 64 && !ldir && !fzs) { fprintf(stderr, "predict: the guided head without one window per k^3 shards needs the full-resolution input: rerun with UFSM_PRED_LAT_INPUT=0\n"); return 2; }
                if (!ldir && !fzs && !(lhm && !ffd) && !lgf) lgf = nn_malloc(ffd ? w3 * 2 * (cfg.lat_guide == 8 ? 19 : cfg.lat_guide == 64 ? 11 : cfg.cout) : w3 * 4 * (rvfuse ? 2 : 1));   /* (fit fields from a latent-grid net: every head at full resolution, fp16) */
                if (!ldir && !fzs && !(lhm && !ffd) && hdiv > 2 && !lrt) lrt = nn_malloc(w3 / 8 * 16);
                nn_ptick("other");
                if (lat_store && (!latc || latc != lz_c)) { fprintf(stderr, "predict: --latents: the store has %d cell channels, the checkpoint %d\n", lz_c, latc); return 2; }
                if (lvq) {   /* a latvol CT: the same from its cells (the first latc of the store's channels) */
                    if (latin) nn_lat_pred_input_qs(lz_d, lv_stride, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, lgain, latc, lvc->step_raw, x2);
                    else nn_lat_input_qs(lz_d, lv_stride, xd, Wz / 4, W / 4, W / 4, lgain, latc, lvc->step_raw, (float)mean, (float)(1.0 / sd), x2);
                } else
                if (lat_store) {   /* the cell channels from the stored coefficients, z-scored with the rebuilt window's statistics */
                    if (latin) nn_lat_pred_input_q(lz_d, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, lgain, latc, (float)lz_sraw, x2);
                    else nn_lat_input_q(lz_d, xd, Wz / 4, W / 4, W / 4, lgain, latc, (float)lz_sraw, (float)mean, (float)(1.0 / sd), x2);
                } else
                if (latc && latin) nn_lat_pred_input_c(ctd, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, lbas, lgain, latc, lstep, x2);
                else if (latc) nn_lat_input_x(xd, W / 4, W / 4, W / 4, lbas, lgain, latc, lstep, x2);   /* the learned cell channels of a --lat-basis checkpoint */
                else if (latin) nn_lat_pred_input(ctd, Wz, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, x2);
                else nn_lowres_input(xd, xs, hdiv, hmode, lrt, x2, h16);
                nn_ptick("input pool");
                const float *l2 = unet_forward_x(u, x2, xn, 0, h16); nn_ptick("trunk forward");
                /* train --fstem: the fine voxel stem on the CT channel (xd channel 0) into the planes the path holds: output o of the stem's list
                   (every non-normal / density head output, in channel order) to plane MAP(c) of its channel c (-1: not held) */
#define PRED_FSTEM(z0_, nz_, planes, ns__, MAP) do { if (fstem_off) break; int fm_[16], k_ = 0; for (int c = 0; c < cfg.cout && k_ < cfg.fstem_out; c++) if (c < 2 || c >= 9) { fm_[k_++] = (MAP); } \
                    if (!xd || !h16 || nn_fstem_fwd(xd, W, W, W, (z0_), (nz_), unet_live_params(u) + unet_fstem_off(u), cfg.fstem, cfg.fstem_out, fm_, (planes), (ns__))) { fprintf(stderr, "predict: the fine stem needs the fp16 input window\n"); return 2; } } while (0)
                static int cq = -1; if (cq < 0) cq = getenv("UFSM_PRED_CELL_Q") != nullptr;
                if (cq && half == 3) cellq_apply(l2, unet_logits_h16(u), cfg.cout, Wz / 4, W / 4, W / 4, hz / 4, halo / 4);
                if (ldir || fzs || (lhm && !ffd)) l2d = l2;
                else if (ffd && cfg.lat_guide == 8) {   /* the general guided head: the 11 task planes, then the 8 coefficient planes (the head's last), the CT terms into the phase channels 0-1 */
                    const size_t cv_ = w3 / 64, es_ = unet_logits_h16(u) ? 2 : 4;
                    if (!h16) { fprintf(stderr, "predict: the guided head needs 16-bit inputs\n"); return 2; }
                    nn_lat_up_slab(l2, unet_logits_h16(u), 11, W / 4, W / 4, W / 4, 0, W, lgf);
                    nn_lat_up_slab((const char *)l2 + (size_t)24 * cv_ * es_, unet_logits_h16(u), 8, W / 4, W / 4, W / 4, 0, W, (char *)lgf + (size_t)11 * w3 * 2);
                    nn_lat_guide4(xd, W, W, W, 0, W, lgf, 11, 0, 2, 0);
                }
                else if (ffd && cfg.lat_guide == 64) {   /* the sub-cell head: the 11 task planes, the phase pair's per-voxel terms */
                    nn_lat_up_slab(l2, unet_logits_h16(u), 11, W / 4, W / 4, W / 4, 0, W, lgf);
                    nn_lat_shuffle(l2, unet_logits_h16(u), W / 4, W / 4, W / 4, 16, 2, 0, W, lgf);
                    if (cfg.fstem) PRED_FSTEM(0, W, lgf, w3, c < 11 ? c : -1);   /* (the 11 task planes) */
                }
                else if (ffd) nn_lat_up_slab(l2, unet_logits_h16(u), cfg.cout, W / 4, W / 4, W / 4, 0, W, lgf);   /* [cout][W][W][W] fp16: the accumulators' layout */
                else if (rvfuse && cfg.lat_guide) {   /* the guided head: the two side logits and their 5 coefficient planes one 32-plane slab at a time (fp16), the CT terms, into [2][W][W][W] */
                    static void *gsl; const size_t hw_ = (size_t)W * W, ns_ = 32 * hw_, cv_ = w3 / 64, es_ = unet_logits_h16(u) ? 2 : 4; const int nb_ = cfg.cout - cfg.lat_guide;
                    if (!gsl) gsl = nn_malloc(10 * ns_ * 2);
                    if (!h16 && cfg.lat_guide != 64) { fprintf(stderr, "predict: the guided head needs 16-bit inputs\n"); return 2; }
                    for (int z0 = 0; z0 < W; z0 += 32) {
                        if (rv2c >= 0) {   /* --rv 2: its interpolated pair */
                            nn_lat_up_slab((const char *)l2 + (size_t)rv2c * cv_ * es_, unet_logits_h16(u), 2, W / 4, W / 4, W / 4, z0, 32, gsl);
                            if (cfg.fstem) PRED_FSTEM(z0, 32, gsl, ns_, c == rv2c || c == rv2c + 1 ? c - rv2c : -1);
                            for (int c = 0; c < 2; c++) nn_d2d_async((char *)lgf + ((size_t)c * w3 + (size_t)z0 * hw_) * 2, (char *)gsl + (size_t)c * ns_ * 2, ns_ * 2);
                            continue;
                        }
                        nn_lat_up_slab(l2, unet_logits_h16(u), 2, W / 4, W / 4, W / 4, z0, 32, gsl);
                        if (cfg.lat_guide == 64) {   /* the sub-cell head: the side pair's per-voxel terms */
                            if (!ufsm_env_on("UFSM_PRED_NOSHUF")) nn_lat_shuffle(l2, unet_logits_h16(u), W / 4, W / 4, W / 4, 16, 2, z0, 32, gsl);   /* (diagnostic: the interpolated base alone) */
                            if (cfg.fstem) PRED_FSTEM(z0, 32, gsl, ns_, c < 2 ? c : -1);
                            for (int c = 0; c < 2; c++) nn_d2d_async((char *)lgf + ((size_t)c * w3 + (size_t)z0 * hw_) * 2, (char *)gsl + (size_t)c * ns_ * 2, ns_ * 2);
                            continue;
                        }
                        nn_lat_up_slab((const char *)l2 + (size_t)(cfg.lat_guide == 8 ? 24 : nb_) * cv_ * es_, unet_logits_h16(u), cfg.lat_guide, W / 4, W / 4, W / 4, z0, 32, (char *)gsl + 2 * ns_ * 2);
                        if (cfg.lat_guide == 8) nn_lat_guide4(xd, W, W, W, z0, 32, gsl, 2, 0, 2, 0); else nn_lat_guide(xd, W, W, W, z0, 32, gsl, 2, 0, 0);
                        for (int c = 0; c < 2; c++) nn_d2d_async((char *)lgf + ((size_t)c * w3 + (size_t)z0 * hw_) * 2, (char *)gsl + (size_t)c * ns_ * 2, ns_ * 2);
                    }
                }
                else nn_lowres_output(l2, unet_logits_h16(u), (shape5){1, rvfuse ? 2 : 1, W / hdiv, W / hdiv, W / hdiv}, hdiv, hmode, (float *)lrt, lgf);   /* (--channel rv on a latent-grid net: both side logits) */
                if (lhm) l2d = l2;   /* (the heads read the cell logits) */
                lg = ldir || fzs || (lhm && !ffd) ? nullptr : lgf; nn_ptick("upsample");
            } else lg = ct_stem ? unet_forward_ct(u, ctd, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, xs, h16)
                                : unet_forward_x(u, xd, xs, 0, h16);
            /* one head of the multi-head path into its device shard (window offset o, shard extent e) */
#define PHEADS_PLACE(oz_, oy_, ox_, ez_, ey_, ex_) do { uint8_t *ds_[PHEAD_MAX]; for (int j_ = 0; j_ < nph; j_++) ds_[j_] = PH[j_].dsh2[phd]; \
                    nn_lat_heads_place(l2d, unet_logits_h16(u), Wz / 4, W / 4, W / 4, nph, ph_c0, ph_c1, ph_sb0, ph_sb1, ph_s0, ph_s1, ds_, \
                    cfg.fstem && !fstem_off ? unet_live_params(u) + unet_fstem_off(u) : nullptr, cfg.fstem, cfg.fstem_out, ct_shared ? ctcp : ctd, (float)mean, (float)(1.0 / sd), nn_get_pred_iso(), \
                    hz, halo, (oz_), (oy_), (ox_), (ez_), (ey_), (ex_), shard); } while (0)
            if (rvfuse && !ldir && !fzs && !lhm) { nn_rv_fuse((void *)lg, w3, half ? cfg.lat_guide != 0 : unet_logits_h16(u)); nn_ptick("rv fuse"); }
            pred_mem_out(u, o);
            if (ctpre && i + 1 < rd.ntiles && atomic_load(&rd.filled) > i + 1) {   /* the next window's CT during this forward (its slot stays held
                                                                                         until that window's statistics, as with the plain upload) */
                if (pre_i >= 0) nn_stream_wait(1, ev_preu);   /* (the previous prefetch's device copy has read ctpre) */
                nn_h2d_copy_stream(ctpre, rd.buf[(i + 1) % NSLOT], w3); nn_event_record(ev_pre, 1); pre_i = i + 1;
            }
            nn_set_logits_h16(half ? ffd != nullptr || (rvfuse && cfg.lat_guide) : unet_logits_h16(u));   /* the readers of lg below */
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
            if (ffzm && !(fea && t->noout)) {   /* (--feather: windows computed only for their spills write no fields) */ /* the window interior's whole cells, finalized into the field slab (flushed when the slab is left) */
                int c0[3], c1[3], nc3[3], ok = 1; const int64_t tw[3] = {tz, ty, tx};
                /* per-shard windows are anchored on the global grid while the shards (and with them the field slabs and rows) start at the
                   box: with a box off the window grid (--grid-origin, (box - origin) % stride != 0) a window's interior covers two shards
                   and the window is run once for each. Its cells are clipped to the tile's shard like its outputs, so each visit writes
                   its shard's cells only: unclipped, the cells of a window across a slab boundary were written past the slab (segfault in
                   ffw_main: wbench DLV -1, a 1024^3 box at z 768 with 640 / 64 windows and --fit-h 4, 2026-11-11). On the grid: no change. */
                const int64_t sh3[3] = {t->shard / (ns[1] * ns[2]), t->shard / ns[2] % ns[1], t->shard % ns[2]};
                for (int d = 0; d < 3; d++) {
                    int64_t lo = tw[d] + hal[d], hi = tw[d] + wn[d] - hal[d]; if (lo < 0) lo = 0; if (hi > bn[d]) hi = bn[d];
                    if (!sup) { const int64_t s0 = sh3[d] * shard, s1 = s0 + shard; if (lo < s0) lo = s0; if (hi > s1) hi = s1; }
                    c0[d] = (int)((lo + ffh - 1) / ffh); c1[d] = (int)(hi / ffh); if (hi == bn[d]) c1[d] = ffF[d];   /* the box's last, possibly partial, cell */
                    nc3[d] = c1[d] - c0[d]; if (nc3[d] <= 0 || nc3[d] > (d ? ffwc : strd[0] / ffh + 1)) ok = 0;
                }
                if (ok && (c1[0] - 1) / FZ != c0[0] / FZ) {   /* (never with the clip above; a clean stop instead of a write past the slab) */
                    fprintf(stderr, "predict: --fit-zarr: a window's cells %d .. %d in z span two field slabs of %d (a box off the window grid?)\n", c0[0], c1[0], FZ); return 1;
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
                        if (fzs) {   /* the chunk's planes a .. b of the window (whole coarse planes around them: the fast upsampling), the accumulators on the slab as a window of its own; the winding's planes with its +-d neighbours */
                            const int a = (int)(cz * ffh - tz), bq = a + lbox[0], zs = a & ~3, ze = (bq + 3) & ~3, nzs = ze - zs, wc = W / 4, lh = unet_logits_h16(u), g8 = cfg.lat_guide == 8, g64 = cfg.lat_guide == 64, cup = g8 || g64 ? 11 : cfg.cout;
                            const size_t hw = (size_t)W * W, ns = (size_t)nzs * hw, cv_ = w3 / 64, es_ = lh ? 2 : 4;
                            if (!fzA) { fzP = ffzc * ffh + 8; fzA = nn_malloc((size_t)(cup + (g8 ? 8 : 0)) * fzP * hw * 2); if (ffw4d) fzB = nn_malloc((size_t)2 * (fzP + 2 * ffw4d) * hw * 2); if (g8 || cfg.fstem) fzX = nn_malloc((size_t)(fzP + 2 * ffw4d + 8) * hw * 2); }
                            if (nzs > fzP || a < hz || bq > Wz - hz) { fprintf(stderr, "predict: --fit-zarr: a z chunk outside its slab (%d .. %d of %d)\n", a, bq, fzP); return 1; }
                            nn_lat_up_slab(l2d, lh, cup, wc, wc, wc, zs, nzs, fzA);
                            if (g64) nn_lat_shuffle(l2d, lh, wc, wc, wc, cfg.cout - 128, 2, zs, nzs, fzA);   /* the sub-cell head: the phase pair's per-voxel terms */
                            /* train --fstem: the fine stem on planes z0 .. z0 + nz of the planes P (channel c at plane MAP), its CT taps (planes z0 - 1 .. z0 + nz) z-scored
                               from the window's bytes into fzX (the stem reads the window as if whole: only those planes) */
#define FZ_STEM(z0_, nz_, P, MAP) do { if (!cfg.fstem || fstem_off) break; const int x0_ = (z0_) > 0 ? (z0_) - 1 : 0, x1_ = (z0_) + (nz_) < Wz ? (z0_) + (nz_) + 1 : Wz; int fm_[16], k_ = 0; \
                                for (int c = 0; c < cfg.cout - (g64 ? 128 : 0) && k_ < cfg.fstem_out; c++) if (c < 2 || c >= 9) { fm_[k_++] = (MAP); } \
                                nn_lat_ct_x(ct_shared ? ctcp : ctd, W, x0_, x1_ - x0_, (float)mean, (float)(1.0 / sd), fzX); \
                                if (nn_fstem_fwd((const void *)((uintptr_t)fzX - (size_t)x0_ * hw * 2), Wz, W, W, (z0_), (nz_), unet_live_params(u) + unet_fstem_off(u), cfg.fstem, cfg.fstem_out, fm_, (P), (size_t)(nz_) * hw)) { \
                                    fprintf(stderr, "predict: the fine stem: slab shape\n"); return 2; } } while (0)
                            FZ_STEM(zs, nzs, fzA, c < 11 ? c : -1);
                            if (g8) {   /* the general guided head: the 8 coefficient planes (the head's last), the CT terms into the phase channels 0-1 */
                                const int x0 = zs > 0 ? zs - 1 : 0, x1 = ze < W ? ze + 1 : W;
                                nn_lat_up_slab((const char *)l2d + (size_t)24 * cv_ * es_, lh, 8, wc, wc, wc, zs, nzs, (char *)fzA + (size_t)11 * ns * 2);
                                nn_lat_ct_x(ct_shared ? ctcp : ctd, W, x0, x1 - x0, (float)mean, (float)(1.0 / sd), fzX);
                                nn_lat_guide4((const void *)((uintptr_t)fzX - (size_t)x0 * hw * 2), W, W, W, zs, nzs, fzA, 11, 0, 2, 0);   /* (the window's x as if whole: only planes x0 .. x1 are read) */
                            }
                            nn_fit_accum_dm(fzA, ct_shared ? nullptr : ctd + (size_t)zs * hw, ct_shared ? ctm + (size_t)zs * hw / 32 : nullptr, nzs, W, 0, halo, (int)(tz - cz * ffh) + zs, (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
                            if (ffw4d) {
                                const int zb = (a - ffw4d) & ~3, zeb = (bq + ffw4d + 3) & ~3, nzb = zeb - zb;
                                nn_lat_up_slab((const char *)l2d + (size_t)ff_w4 * cv_ * es_, lh, 2, wc, wc, wc, zb, nzb, fzB);
                                FZ_STEM(zb, nzb, fzB, c == ff_w4 ? 0 : c == ff_w4 + 1 ? 1 : -1);   /* (the winding mod 4 pair has the stem's term too) */
                                nn_fit_accum_w4(fzB, ct_shared ? nullptr : ctd + (size_t)zb * hw, ct_shared ? ctm + (size_t)zb * hw / 32 : nullptr, nzb, W, ffw4d, halo, (int)(tz - cz * ffh) + zb, (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, 0, ffw4d, ffacc4);
                            }
                        } else {
                        nn_fit_accum_dm(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_rec, ff_ph, ff_nrm, ff_den, ffds, ffdc, ffacc);
                        if (ffw4d) nn_fit_accum_w4(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, ff_w4, ffw4d, ffacc4);
                        }
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
            if (ffd && out_ch && lg) lg = (const float *)((const char *)lg + (size_t)out_ch * w3 * (unet_logits_h16(u) ? 2 : 4));   /* every head computed: channel out_ch of the [c][window] logits for the probability volume (e.g. --channel ink beside --fit-zarr) */
            if (sheet_task) lg += (size_t)out_ch * w3;   /* channel c of the [c][W^3] logits (B = 1); otherwise the only one */
            if (!ffd && tta > 1) {   /* --tta on the probability: the mean of the K passes' probabilities, as logit(mean) in the last pass's buffer */
                static void *tacc; if (!tacc && !(tacc = nn_malloc_try(w3 * 2))) { fprintf(stderr, "predict: --tta: no device memory for the accumulator (%.2f GB)\n", w3 * 2 / 1e9); return 1; }
                const int lh0 = unet_logits_h16(u);
                nn_tta_pacc(lg, lh0, w3, tacc, 1);
                for (int p = 1; p < tta; p++) { lg = tta_pass(&TC, tta_order[p]); nn_tta_pacc(lg, unet_logits_h16(u), w3, tacc, 0); }
                nn_tta_pmean((void *)lg, unet_logits_h16(u), w3, tacc, tta);
                nn_set_logits_h16(unet_logits_h16(u));
            }
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
                    if (lhm) {   /* every head into its own shard of this super-shard */
                        for (int j = 0; j < nph; j++) phead_fresh(&PH[j], phd);
                        PHEADS_PLACE((int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]));
                        for (int j = 0; j < nph; j++) if (phead_submit(&PH[j], phd, qz, qy, qx, SZ * kss)) return 1;
                        phd ^= 1;
                        nn_ptick("heads place + d2h");
                    }
                    if (WR.skip) { if (writer_submit(&WR, sb, qz, qy, qx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb = writer_next(&WR, sb); sbuf = WR.buf[sb]; continue; }   /* --no-out */
                    nn_zero(dsh, WR.bytes);
                    if (ldir && cfg.lat_guide == 64) nn_lat_place_rvs(l2d, unet_logits_h16(u), 16, W / 4, W / 4, W / 4, ct_shared ? nullptr : ctd, ct_shared ? ctmi : nullptr, nn_get_pred_iso(), halo, halo,
                                              (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else if (ldir && cfg.lat_guide) nn_lat_place_rvg(l2d, unet_logits_h16(u), cfg.cout - cfg.lat_guide, W / 4, W / 4, W / 4, ct_shared ? ctcp : ctd, (float)mean, (float)(1.0 / sd), ct_shared ? nullptr : ctd, ct_shared ? ctmi : nullptr, nn_get_pred_iso(), halo, halo,
                                              (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else if (ldir) nn_lat_place_rv(l2d, unet_logits_h16(u), W / 4, W / 4, W / 4, ct_shared ? nullptr : ctd, ct_shared ? ctmi : nullptr, nn_get_pred_iso(), halo, halo,
                                              (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else if (ct_shared) nn_pred_place_m(lg, ctmi, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else nn_pred_place(lg, ctd, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    nn_ptick("place kernels");
                    writer_d2h(&WR, sb, dsh);
                    nn_ptick("shard d2h");
                    if (writer_submit(&WR, sb, qz, qy, qx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
                    sb = writer_next(&WR, sb); sbuf = WR.buf[sb];
                    nn_ptick("writer wait");
                }
                const char *e = nn_check();
                if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
                if (pprof) { nn_sync(); p_gpu += now() - pt; }
                nn_ptick("other");
                ntiles++;
                fprintf(stderr, "\rsuper-shard %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, rd.ntiles, ntiles, nskip, now() - t0);
                continue;
            }
            if (gpath) {   /* interior straight into the device shard buffer (--no-out: nothing placed) */
                if (lhm) PHEADS_PLACE((int)(tz - so[0]), (int)(ty - so[1]), (int)(tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]));
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
    if (nn_ptick_on()) { fprintf(stderr, "\n"); nn_ptick_report((int)(ntiles > 0 ? ntiles : 1)); }
    if (pprof && rd.dbuf[0]) fprintf(stderr, "\npredict profile: device CT windows: %ld chunks all on the GPU, %ld with host substreams, %ld fill\n", (long)rd.ndev, (long)rd.nhost, (long)rd.nskip);
    if (pprof) fprintf(stderr, "\npredict profile (s): main: wait-reader %.2f h2d %.2f gpu(fwd+d2h) %.2f interior-copy %.2f writer-wait %.2f fit-writer-wait %.2f | reader: read %.2f stats %.2f slot-wait %.2f | wall %.2f, %ld tiles\n",
                       p_wait, p_h2d, p_gpu, p_copy, p_wr, p_ffw, rd.t_read, rd.t_stats, rd.t_wait, now() - t0, ntiles);
    if (!sup && cur_shard >= 0 && any && lhm) for (int j = 0; j < nph; j++) if (phead_submit(&PH[j], phd, sz, sy, sx, sz)) return 1;
    if (!sup && cur_shard >= 0 && any) { if (gpath && !WR.skip) writer_d2h(&WR, sb, dsh); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } }
    else if (!sup && cur_shard >= 0 && WR.donefd >= 0) dprintf(WR.donefd, "%lld %lld %lld\n", (long long)(sz + soff[0]), (long long)(sy + soff[1]), (long long)(sx + soff[2]));
    if (sheet_task && cur_shard>=0 && sheet_write_shard(out,sheet_shard,shard,sz,sy,sx)) return 1;
    if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
    { const double tf = now();
      for (int j = 0; j < nph; j++) { if (writer_finish(PH[j].wr)) { fprintf(stderr, "predict: head %s: %s\n", PH[j].name, z3w_error()); return 1; }
                                      if (PH[j].wr->donefd >= 0) close(PH[j].wr->donefd); }
      if (gpyr) { const double tf = now(); double te = 0; long ne = 0;
                  for (int j = 0; j < nph; j++) { te += PH[j].pya->t_enc; ne += PH[j].pya->n_enc; if (pyacc_finish(PH[j].pya)) { fprintf(stderr, "predict: head %s: the GPU pyramid failed\n", PH[j].name); return 1; } }
                  if (pprof) fprintf(stderr, "predict profile (s): GPU pyramid: %.2f s in the head writers (%ld chunks of levels >= 1 encoded as they came), %.2f s for the rest\n", te, ne, now() - tf); }
      if (writer_finish(&WR)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
      if (pprof && !lhm) fprintf(stderr, "predict profile (s): writer: %.2f thread-seconds compressing %ld shards (%.0f MB/s per writer thread), tail after the last window %.2f\n",
                         WR.t_write, WR.n_write, WR.t_write > 0 ? WR.n_write * (double)WR.bytes / WR.t_write / 1e6 : 0., now() - tf);
      if (pprof && lhm) { double tw = 0; long nw = 0; for (int j = 0; j < nph; j++) { tw += PH[j].wr->t_write; nw += PH[j].wr->n_write; }
                          fprintf(stderr, "predict profile (s): %d head writers: %.2f s compressing %ld shards (%.0f MB/s per writer), tail after the last window %.2f\n",
                                  nph, tw, nw, tw > 0 ? nw * (double)WR.bytes / tw / 1e6 : 0., now() - tf); } }
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
    if (pf_on) { atomic_store(&PF.stop, 1); pthread_mutex_lock(&rd.mu); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu); pthread_join(pfth, nullptr);
                 if (pprof) fprintf(stderr, "predict profile (s): prefetch: %ld chunks fetched ahead of the reader in %.2f s (%d windows ahead, %d threads)%s\n", PF.nfetch, PF.t_fetch, PF.ahead, PF.nt, atomic_load(&PF.bad) ? ", some failed (the reader fetched them)" : ""); }
    pthread_join(rth, nullptr);
    for (int k = 0; k < NSLOT; k++) { if (gpath) nn_host_free(rd.buf[k]); else free(rd.buf[k]); }
    for (int k = 0; k < NSLOT; k++) if (rd.dbuf[k]) nn_free(rd.dbuf[k]);
    for (int t = 0; t < rd.nctw; t++) nn_ctw_free(rd.ctw[t]);
    free(rd.ctw);
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
    if (pp) {   /* level 1: the pipelined layers, then the rest */
        pthread_mutex_lock(&pp->mu); pp->stop = 1; pthread_cond_broadcast(&pp->cv); pthread_mutex_unlock(&pp->mu);
        const double tw = now(); pthread_join(pp->th, nullptr); const double wait = now() - tw;
        if (pp->failed) return 1;
        long nd = 0; for (int64_t z = 0; z < pp->nz1; z++) nd += pp->done[z];
        const double tr = now();
        for (int64_t z = 0; z < pp->nz1;) {
            if (pp->done[z]) { z++; continue; }
            int64_t e = z; while (e < pp->nz1 && !pp->done[e]) e++;
            if (pyramid_build_level_z(out, um_l, 1, ash, shard, q, 0, nthreads, attrs, z, e, 1)) return 1;
            z = e;
        }
        fprintf(stderr, "level 1: %ld of %lld shard layers built behind the windows (%.0f s), %.0f s waited, %.0f s for the rest\n", nd, (long long)pp->nz1, pp->busy, wait, now() - tr);
        free(pp->done); free(pp->jq);
    }
    if (hspec) {   /* the heads' stores (the main store holds nothing: removed) */
        /* the heads' pyramids in parallel (each decodes its level 0 back and pools it on the CPU: ~2.4 s per head and 19 Gvox on the pod one after another) */
        const double tp = now(); pthread_t pth[PHEAD_MAX]; phead_pyr pj[PHEAD_MAX];
        if (gpyr) { for (int j = 0; j < nph; j++) { z3w_close(PH[j].w); if (pyramid_write_group(PH[j].dir, um_l, PH[j].pya->nlev, "ufsm-recto", PH[j].attrs)) return 1; } }
        else for (int j = 0; j < nph; j++) { z3w_close(PH[j].w); pj[j] = (phead_pyr){.h = &PH[j], .um = um_l, .ash = ash, .shard = shard, .q = q, .nlev = nlev, .nthreads = nthreads}; pthread_create(&pth[j], nullptr, phead_pyr_main, &pj[j]); }
        int pbad = 0; if (!gpyr) for (int j = 0; j < nph; j++) { pthread_join(pth[j], nullptr); pbad |= pj[j].fail; }
        if (pbad) return 1;
        fprintf(stderr, "predict: %d head stores under %s (pyramids %.1f s)\n", nph, out, now() - tp);
        char p[1600]; snprintf(p, sizeof p, "%s/zarr.json", ldir); unlink(p); rmdir(ldir);
    } else {
    if (!gpyr) for (int l = pp ? 2 : 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, ash, shard, q, 0, nthreads, attrs)) return 1;
    if (pyramid_write_group(out, um_l, gpyr ? PH[0].pya->nlev : nlev, "ufsm-recto", attrs)) return 1;
    }
    fprintf(stderr, "wrote %s (%ld tiles in %.0fs)\n", out, ntiles, now() - t0);
    if (lvm) { double ds = 0; long dn = 0; nn_latvol_stats(lvL, &ds, &dn);
        fprintf(stderr, "predict: latvol CT: %.3f GB of chunk streams uploaded and decoded on the device (%.4f B per box voxel); the decode kernels %.3f s in %ld windows (%.1f Gvox/s of window voxels)\n",
                lv_up / 1e9, lv_up / ((double)bn[0] * bn[1] * bn[2]), ds, dn, ds > 0 ? (double)dn * (double)w3 / 1e9 / ds : 0.); }
    if (ufsm_env_on("UFSM_PRED_IO")) {   /* reads per store root and the process's read bytes per box voxel (--latents against the CT path) */
        store_io_report(stderr);
        long long rch = 0, rdb = 0; char ln[128]; FILE *io = fopen("/proc/self/io", "r");
        while (io && fgets(ln, sizeof ln, io)) { if (!strncmp(ln, "rchar:", 6)) rch = atoll(ln + 6); else if (!strncmp(ln, "read_bytes:", 11)) rdb = atoll(ln + 11); }
        if (io) fclose(io);
        const double gv = (double)bn[0] * bn[1] * bn[2] / 1e9, dt = now() - t0;
        fprintf(stderr, "predict io: box %.4f Gvox in %.2f s (%.3f Gvox/s); read %.4f GB (%.4f GB per Gvox), from disk %.4f GB (%.4f GB per Gvox)%s\n", gv, dt, gv / dt,
                rch / 1e9, rch / 1e9 / gv, rdb / 1e9, rdb / 1e9 / gv, lat_store ? ", --latents" : "");
        if (lat_store) fprintf(stderr, "predict io: the latent store: %.4f GB of chunks and indices read (%.4f GB per Gvox)\n", zr_bytes_read() / 1e9, zr_bytes_read() / 1e9 / gv);
    }
    z3_close(ct); store_close(s); unet_free(u);
    free(sheet_window); free(sheet_shard); free(sheet_rows); sheet_free(sheet);
    return 0;
}
