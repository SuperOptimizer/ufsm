/* ufsm predict: sliding-window inference of a checkpoint over a CT box, written as a volcomp zarr v3
   probability pyramid (uint8 p*255, q=8) in the published layout. */
#include "nn.h"
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
#include "sweep.h"
#include "z3w.h"
#include "zarr3.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* tile read-ahead: a reader thread fills a ring of CT windows (with their nonzero count and intensity sums) in tile
   order while the main thread runs the network on the previous one */
#define NSLOT 3
typedef struct { int64_t tz, ty, tx; int64_t shard; } tile_t;
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
/* shard writer: the main thread hands a filled shard buffer to a writer thread and continues into the other buffer */
typedef struct { z3w *w; int nthreads; size_t bytes; uint8_t *buf[2]; int64_t sz[2], sy[2], sx[2]; int pending[2]; atomic_int failed; pthread_mutex_t mu; pthread_cond_t cv; int stop; } writer_t;
static void *writer_main(void *arg) {
    writer_t *W = arg;
    for (;;) {
        pthread_mutex_lock(&W->mu);
        int k = -1;
        while (k < 0 && !W->stop) { for (int i = 0; i < 2; i++) if (W->pending[i]) { k = i; break; } if (k < 0) pthread_cond_wait(&W->cv, &W->mu); }
        if (k < 0) { pthread_mutex_unlock(&W->mu); return nullptr; }
        pthread_mutex_unlock(&W->mu);
        if (z3w_write_shard(W->w, W->sz[k], W->sy[k], W->sx[k], W->buf[k], W->nthreads)) atomic_store(&W->failed, 1);
        pthread_mutex_lock(&W->mu); W->pending[k] = 0; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
}
/* queue buffer k for writing; returns when the OTHER buffer is free to fill */
static int writer_submit(writer_t *W, int k, int64_t sz, int64_t sy, int64_t sx) {
    pthread_mutex_lock(&W->mu);
    W->sz[k] = sz; W->sy[k] = sy; W->sx[k] = sx; W->pending[k] = 1; pthread_cond_broadcast(&W->cv);
    while (W->pending[k ^ 1]) pthread_cond_wait(&W->cv, &W->mu);
    pthread_mutex_unlock(&W->mu);
    return atomic_load(&W->failed);
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
static const char *ffz_names[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
static const int ffz_fill[10] = {128, 128, 0, 0, 0, 0, 0, 0, 0, 0};
/* --fit-zarr: write the slab buffer's field shards (z index fsz) of every array */
/* host field slab slab (FZ cells in z from slab FZ, FZ a multiple of the field shard FS; planes past F[0] not held) -> its rows of
   shards, the shards compressed and written on threads (a 128^3 field shard is one chunk, so z3w_write_shard alone is one thread:
   the 10 arrays of a 256 x 2048 x 2048 box took 163 s) */
typedef struct { z3w *const *zw; uint8_t *const *sl; int64_t slab; const int *F; int FS, FZ, nq, ny, nx; _Atomic int next, fail; } ffz_job;
static void *ffz_worker(void *a) {
    ffz_job *J = a; const int FS = J->FS, *F = J->F; uint8_t *sb = malloc((size_t)FS * FS * FS);
    const int njob = 10 * J->nq * J->ny * J->nx;
    for (;;) {
        const int t = atomic_fetch_add(&J->next, 1); if (t >= njob || atomic_load(&J->fail)) break;
        const int k = t / (J->nq * J->ny * J->nx), q = t / (J->ny * J->nx) % J->nq, fsy = t / J->nx % J->ny, fsx = t % J->nx;
        const int64_t fsz = J->slab * (J->FZ / FS) + q;
        memset(sb, ffz_fill[k], (size_t)FS * FS * FS);
        for (int a2 = 0; a2 < FS && fsz * FS + a2 < F[0]; a2++) for (int b = 0; b < FS && (int64_t)fsy * FS + b < F[1]; b++) {
            const int64_t x0 = (int64_t)fsx * FS, nx = F[2] - x0 < FS ? F[2] - x0 : FS;
            memcpy(sb + ((size_t)a2 * FS + b) * FS, J->sl[k] + ((size_t)(q * FS + a2) * F[1] + (size_t)fsy * FS + b) * F[2] + x0, (size_t)nx);
        }
        if (z3w_write_shard(J->zw[k], fsz, fsy, fsx, sb, 1)) atomic_store(&J->fail, 1);
    }
    free(sb); return nullptr;
}
static int ffz_flush(z3w *const zw[10], uint8_t *const sl[10], int64_t slab, const int F[3], int FS, int FZ) {
    int nq = 0; for (int q = 0; q < FZ / FS; q++) if ((slab * (FZ / FS) + q) * FS < F[0]) nq = q + 1;
    ffz_job J = {.zw = zw, .sl = sl, .slab = slab, .F = F, .FS = FS, .FZ = FZ, .nq = nq, .ny = (F[1] + FS - 1) / FS, .nx = (F[2] + FS - 1) / FS};
    atomic_store(&J.next, 0); atomic_store(&J.fail, 0);
    long nt = sysconf(_SC_NPROCESSORS_ONLN); if (nt > 64) nt = 64; if (nt < 1) nt = 1;
    pthread_t th[64]; int up[64] = {0};
    for (int i = 1; i < nt; i++) up[i] = pthread_create(&th[i], nullptr, ffz_worker, &J) == 0;
    ffz_worker(&J);
    for (int i = 1; i < nt; i++) if (up[i]) pthread_join(th[i], nullptr);
    if (atomic_load(&J.fail)) return -1;
    const int64_t held = F[0] < FZ ? F[0] : FZ;
    for (int k = 0; k < 10; k++) memset(sl[k], ffz_fill[k], (size_t)held * F[1] * F[2]);
    return 0;
}

static int writer_finish(writer_t *W, pthread_t th) {
    pthread_mutex_lock(&W->mu); while (W->pending[0] || W->pending[1]) pthread_cond_wait(&W->cv, &W->mu); W->stop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    pthread_join(th, nullptr); return atomic_load(&W->failed);
}
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
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
typedef struct { unet *u; uint8_t *ctd, *dsh, *own; uint32_t *ctm; void *stsc; float *dyd, *rows; double *red; int cuda_bad; } pside_t;
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
    J->sb ^= 1;
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
        const float *lg = unet_forward_ct(S->u, S->ctd, (float)mean, (float)(1.0 / sd), S->dyd, S->dyd + Dl, J->ax->n > 0 ? J->sign : 0, xs, J->h16);
        nn_set_logits_h16(unet_logits_h16(S->u));   /* (both halves build alike: the process-wide flag holds the same value) */
        if (J->kss > 1) {   /* every shard of this super-shard from the one window */
            const int64_t SZ = t->shard / (J->nss[1] * J->nss[2]), SY = (t->shard / J->nss[2]) % J->nss[1], SX = t->shard % J->nss[2], k = J->kss;
            for (int64_t qz = SZ * k; qz < SZ * k + k && qz < J->ns[0]; qz++) for (int64_t qy = SY * k; qy < SY * k + k && qy < J->ns[1]; qy++)
            for (int64_t qx = SX * k; qx < SX * k + k && qx < J->ns[2]; qx++) {
                const int64_t qo[3] = {qz * shard, qy * shard, qx * shard}, qe[3] = {qo[0] + shard < J->bn[0] ? qo[0] + shard : J->bn[0], qo[1] + shard < J->bn[1] ? qo[1] + shard : J->bn[1], qo[2] + shard < J->bn[2] ? qo[2] + shard : J->bn[2]};
                nn_zero(S->dsh, sh3); memset(S->own, 0, (size_t)shard);
                nn_pred_place_mz(lg, S->ctm, W, halo, olo, ohi, zoff, (int)(t->tz - qo[0]), (int)(t->ty - qo[1]), (int)(t->tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, S->dsh);
                psplit_mark(S->own, shard, t->tz - qo[0], qe[0] - qo[0], ilo, ihi);
                psplit_flush(J, side, qz, qy, qx);
            }
        } else {
            nn_pred_place_mz(lg, S->ctm, W, halo, olo, ohi, zoff, (int)(t->tz - so[0]), (int)(t->ty - so[1]), (int)(t->tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]), shard, S->dsh);
            psplit_mark(S->own, shard, t->tz - so[0], se[0] - so[0], ilo, ihi);
            any = 1;
        }
        if (nn_check()) S->cuda_bad = 1;
        if (!side) { J->ndone++; fprintf(stderr, "\rwindow %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, J->ntiles, J->ndone, J->nskip, now() - J->t0); }
    }
    if (J->kss == 1 && cur >= 0 && any) psplit_flush(J, side, sz, sy, sx);
}

int cmd_predict(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm predict <ckpt> <root> <ct-group-key> <out-dir> --um U [--level 0] [--box z,y,x,nz,ny,nx] [--window W] [--act-mx8 1]\n"
                        "       [--halo H] [--shard 512] [--gpu 0] [--cache DIR] [--axis umbilicus.json] [--levels 4] [--q 8] [--threads 16]\n"
                        "       [--prec 0|1|2|3|4] [--policy enc0=1,...] [--f16 0|1] [--fp4 0|1] [--input-mx 0|1] [--input-prec 0|4|8] [--ema 1]   EMA or current weights (0)\n"
                        "       [--gn-stats stored|legacy]   defaults to the checkpoint's normalization contract\n"
                        "       [--reference reference.json]   required for a surface_winding checkpoint, native level 0\n"
                        "       [--grid-origin z,y,x]   anchor tile interiors in selected-level CT voxels, independent of the output box\n"
                        "       [--gpus 0,1]   one worker per GPU over the shards of the same output (needs --box)\n"
                        "       [--split z --gpus 0,1]   both GPUs on every window, one z half each (window a multiple of 2^levels; default window shard + 128, halo 64)\n"
                        "       [--channel 0]   output channel written (band_affinity checkpoints: 1..6 = affinities z1 y1 x1 z8 y8 x8)\n"
                        "       [--fit-fields DIR] [--fit-h 8]   task unwrap checkpoints: also the winding fitter's input fields of the box (ufsm fit DIR)\n"
                        "       [--fit-zarr DIR]                 the same fields as zarr arrays, flushed slab by slab (whole volumes)\n"
                        "       [--radial-sign 1|-1]   -1 reverses the direction input (the CT is unchanged): a recto model then marks the verso face\n"
                        "       [--gn-save FILE]   pool the GroupNorm statistics of every window processed into FILE (calibration)\n"
                        "       [--gn-frozen FILE]   use FILE's pooled GroupNorm statistics instead of each window's (no dependence on the window)\n"
                        "       [--papyrus-cut T]   task papyrus nets conditioned on the cutoff: papyrus where the smoothed CT exceeds T (raw CT units)\n");
        fprintf(stderr, "  Default window: shard + 32 with halo 16 (one window per shard) when it fits the GPU, else 288 with halo 8. For exact shard tiling, window - 2*halo should divide the shard size: 544 with halo 16 gives 512; 288 with halo 16 gives 256. Validate window and halo with the checkpoint.\n"
                        "  Window - 2*halo = k * shard (k >= 2) with the box on that grid computes one window per k^3 shards (big GPUs: fewer, larger windows).\n"
                        "  Inference memory: UFSM_INFER_SLAB=N (default 64) runs the level-0/1 convs in place in N-plane slabs; UFSM_INFER_OFFLOAD=1 (default) keeps\n"
                        "  the coarse levels in the level-0 buffer's tail while it waits on the host: ~21.3 B per window voxel (0: ~27; slabs 0: ~43.5).\n");
        return 2;
    }
    const char *ckpt = argv[2], *root = argv[3], *key = argv[4], *out = argv[5];
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), W = atoi(opt(argc, argv, "--window", "288")), halo = atoi(opt(argc, argv, "--halo", "8"));
    int shard = atoi(opt(argc, argv, "--shard", "512")), gpu = atoi(opt(argc, argv, "--gpu", "0")), nlev = atoi(opt(argc, argv, "--levels", "4"));
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
    int nthreads = atoi(opt(argc, argv, "--threads", "16"));
    const int use_ema = atoi(opt(argc, argv, "--ema", "1")) != 0;
    const double radial_sign = atof(opt(argc, argv, "--radial-sign", "1"));   /* -1: reversed direction input (a recto model then marks the verso face) */
    if (radial_sign != 1 && radial_sign != -1) { fprintf(stderr, "predict: --radial-sign must be 1 or -1\n"); return 2; }
    const int out_ch = atoi(opt(argc, argv, "--channel", "0"));   /* output channel to write (band_affinity: 1..6 = affinities) */
    /* --fit-fields DIR (task unwrap checkpoints: recto 0, ink 1, phase 2-3, normal 4-9, density 10): every head computed, each
       window's interior pooled on the device into the winding fitter's fields (h native voxels per field voxel, --fit-h 8) of the
       box, written to DIR at the end (src/fit.h fit_fields_save). One GPU, the CT window in its own buffer (no aliasing). */
    const char *ffd = opt(argc, argv, "--fit-fields", nullptr); const int ffh = atoi(opt(argc, argv, "--fit-h", "8"));
    const char *ffz = opt(argc, argv, "--fit-zarr", nullptr);   /* the same fields of a whole volume: 10 uint8 zarr arrays, flushed slab by slab */
    if (ffd && ffz) { fprintf(stderr, "predict: --fit-fields or --fit-zarr\n"); return 2; }
    if ((ffd || ffz) && (ffh < 1 || ffh > 32)) { fprintf(stderr, "--fit-h 1..32\n"); return 2; }   /* (inference aliasing stays on: the fit pooling reads
                                                                                                       the material mask from the CT bit plane; it was off: ~80 instead of ~45 B per window voxel) */
    if (ffz) ffd = ffz;   /* the shared checks and settings below */
    const int ffzm = ffz != nullptr;
    float q = (float)atof(opt(argc, argv, "--q", "8"));
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
    if (out_ch < 0 || out_ch >= nout) { fprintf(stderr, "predict: --channel %d outside the checkpoint's %d outputs\n", out_ch, nout); return 2; }
    const char *sweep_out = opt(argc, argv, "--cond-sweep", nullptr);   /* train --cond checkpoints: conditioned inference (src/sweep.h) */
    if (sweep_out && (cfg.cin != 7 || cfg.cout < 15 || half || sheet_task || ffd)) { fprintf(stderr, "predict: --cond-sweep needs a train --cond checkpoint (7 inputs, 15 outputs), no --fit-fields\n"); return 2; }
    if (ffd && (cfg.cout < 11 || half || sheet_task)) { fprintf(stderr, "predict: --fit-fields needs a task unwrap checkpoint (11+ outputs) at full resolution\n"); return 2; }
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
    if (W % (hdiv << (cfg.nlev - 1))) { fprintf(stderr, "window must be divisible by %d\n", hdiv << (cfg.nlev - 1)); return 2; }
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
            char attrs[2000]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"gpus\":%d,\"ema\":%d%s}}", content, ckpt, ngpu, use_ema, grid_attrs);
            for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
            if (pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs)) return 1;
            fprintf(stderr, "wrote %s (%d GPUs)\n", out, ngpu);
            return 0;
        }
    }
    if (nn_init(gpu)) { fprintf(stderr, "cannot select GPU %d\n", gpu); return 1; }
    unet *u = unet_create(&cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "cannot load %s\n", ckpt); return 1; }
    unet_use_ema(u, use_ema);
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
        unet_use_ema(u1, use_ema);
        nn_init(gpu);
    }
    for (int k = 0; k < 1 + split; k++) {
        unet *uk = k ? u1 : u;
        if (k) nn_init(gpus[1]);
        if (gnh) unet_gn_freeze(uk, gnh);
        if (ffd || sweep_out) unet_set_head_channels(uk, 0, cfg.cout);   /* every head: the recto channel 0 is written, the others pooled (or swept) */
        /* fp16 logits (MX activations): the readers below take them as they are (nn_set_logits_h16 after each forward); the 11 heads
           of --fit-zarr were 44 B per window voxel in fp32, which pushed 640^3 windows past the 16 GB card (paged: 40x slower).
           UFSM_PRED_LOGITS32=1 keeps fp32 */
        if (!sheet_task && !half && !sweep_out && !ufsm_env_on("UFSM_PRED_LOGITS32")) unet_set_fp16_logits(uk, 1);
        else if (!sheet_task) unet_set_head_channels(uk, hmode == 2 ? 8 * out_ch : out_ch, hmode == 2 ? 8 : 1);   /* only the written logit channel (desk widths: -24 B per window voxel) */
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
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    { const char *b = opt(argc, argv, "--box", nullptr); long long v[6]; if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } } }
    const int H0 = 1 << (cfg.nlev - 1);   /* --split: halo planes of each half */
    if (split && !win_given) { W = shard + 128; halo = 64; fprintf(stderr, "predict: window %d, halo %d (split default; --window / --halo set them; window - 2 halo = k * shard computes one window per k^3 shards)\n", W, halo); }
    if (split && W % (2 * H0)) { fprintf(stderr, "--split z: the window must be a multiple of %d\n", 2 * H0); return 2; }
    if (!split && !win_given && !sheet_task && bn[0] >= shard && bn[1] >= shard && bn[2] >= shard) {   /* default window: one per shard (shard + 2 x 16 halo) for boxes of whole shards when it fits this GPU, else 288 / 8.
                                          Desk widths, 512 shards, laptop: 16% less GPU time per shard than 8 windows of 288, peak 9.4 GB */
        const int Wc = shard + 32, div = 1 << (cfg.nlev - 1);
        if (Wc % (hdiv * div) == 0) {
            const size_t wc3 = (size_t)Wc * Wc * Wc, need = (half ? unet_infer_bytes(u, (shape5){1, cfg.cin, Wc / hdiv, Wc / hdiv, Wc / hdiv}) + 14 * wc3 : unet_infer_bytes(u, (shape5){1, 4, Wc, Wc, Wc}))
                                                     + wc3 + wc3 / 8 + (size_t)shard * shard * shard + ((size_t)1 << 30);   /* (+ the CT window, or its mask when it lives in the network's skip buffer) */
            if (need <= nn_mem_free()) { W = Wc; halo = 16; }
        }
        fprintf(stderr, "predict: window %d, halo %d (default: one window per %d shard when it fits, else 288 / 8, 288 / 16 with --fit-zarr; --window / --halo set them)\n", W, halo, shard);
        if (box) for (int d = 0; d < 3; d++) if (boxv[d + 3] > INT64_MAX - W || boxv[d] > INT64_MAX - W - boxv[d + 3]) { fprintf(stderr, "prediction box exceeds the coordinate range\n"); return 2; }
    }
    if (ffz && !win_given && !split && W == 288 && halo == 8) halo = 16;   /* --fit-zarr fallback: a z stride (256) dividing the shard */
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
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"step\":%d,\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d,\"window_z\":%d,\"halo_z\":%d,\"ema\":%d%s}}",
             content, ckpt, step, root, ak, (long long)bo[0], (long long)bo[1], (long long)bo[2], W, halo, Wz, hz, use_ema, grid_attrs);
    z3w *w = z3w_create(ldir, bn, shard, q, 0, attrs);
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
    /* default path (UFSM_PRED_HOSTPATH=1: the previous host path, for A/B): pinned reader / shard buffers, window statistics
       on the device, the interior probabilities placed into a device shard buffer that is downloaded once per shard */
    const int gpath = !ufsm_env_on("UFSM_PRED_HOSTPATH");
    WR.buf[0] = gpath ? nn_host_alloc(WR.bytes) : malloc(WR.bytes); WR.buf[1] = gpath ? nn_host_alloc(WR.bytes) : malloc(WR.bytes);
    uint8_t *dsh = gpath && !split ? nn_malloc(WR.bytes) : nullptr; void *stsc = nn_malloc(64); pthread_mutex_init(&WR.mu, nullptr); pthread_cond_init(&WR.cv, nullptr);
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
    uint8_t *ctd = nullptr; uint32_t *ctm = nullptr; void *xd = nullptr; int ct_shared = 0, xd_shared = 0;
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
            if (!unet_ct_stem(uk) || !S->ctd) { fprintf(stderr, "--split z needs the CT-stem build (UFSM_INFER_SLAB > 0, MX-fp8 stem input)\n"); return 2; }
            S->ctm = nn_malloc((l3 + 31) / 32 * 4); S->dsh = nn_malloc(WR.bytes); S->stsc = nn_malloc(64); S->red = nn_malloc(4 * sizeof(double));
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
        ctm = ct_shared ? nn_malloc((w3 + 31) / 32 * 4) : nullptr;
        xd = ct_stem || half || auxz || cfg.cin == 7 ? nullptr : h16 && !sheet_task ? unet_input_scratch(u, 4 * w3 * 2) : nullptr;
        xd_shared = xd != nullptr;
        if (!xd && !ct_stem) xd = nn_malloc((size_t)xs.c * w3 * (h16 ? 2 : 4));
        if (auxz) { auxd = nn_malloc(w3); auxh = malloc(w3); }
        if (half) { x2 = nn_malloc(shape_numel(xn) * (h16 ? 2 : 4)); lgf = nn_malloc(w3 * 4); if (hdiv > 2) lrt = nn_malloc(w3 / 8 * 16); }
        if (ct_stem) fprintf(stderr, "predict: in-place slab inference from the CT window (%.1f B per window voxel)\n", (double)unet_activation_bytes(u) / w3);
    }
    int64_t ns[3];
    for (int d = 0; d < 3; d++) ns[d] = (bn[d] + shard - 1) / shard;
    /* interior = k whole shards (k >= 2, the tile grid on the shard grid): one window per k^3 shards, each shard placed from it and
       written (INFERENCE_MEMORY_PLAN.md, memory into speed); otherwise one or more windows per shard, clipped to it */
    int kss = !slab && !sheet_task && gpath && stride % shard == 0 && stride / shard >= 2 ? stride / shard : 1;
    for (int d = 0; d < 3 && kss > 1; d++) { int64_t ph = grid ? (bo[d] - grid_origin[d]) % stride : 0; if (ph < 0) ph += stride; if (ph) kss = 1; }
    int64_t nss[3]; for (int d = 0; d < 3; d++) nss[d] = (ns[d] + kss - 1) / kss;
    float *ffacc = nullptr; int ffbox[3], ffF[3]; size_t ffn = 0;
    z3w *ffzw[10] = {0}; uint8_t *ffsl[10] = {0}; const int FFS = 128; int FZ = FFS; int64_t ffslab = 0; uint8_t *ff8d = nullptr, *ff8h = nullptr; size_t ffwn = 0; int ffwc = 0, ffzc = 1;
    if (ffd) {
        for (int d = 0; d < 3; d++) { ffbox[d] = (int)bn[d]; ffF[d] = (int)((bn[d] + ffh - 1) / ffh); }
        if (!ffzm) {   /* the whole box on the device */
            ffn = (size_t)ffF[0] * ffF[1] * ffF[2]; ffacc = nn_malloc(12 * ffn * 4);
            if (!ffacc) { fprintf(stderr, "predict: no device memory for the fit fields\n"); return 1; }
            nn_zero(ffacc, 12 * ffn * 4);
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
            char at[512]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h)\",\"h\":%d,\"origin_zyx\":[%lld,%lld,%lld],\"checkpoint\":\"%s\"}}", ffh, (long long)bo[0], (long long)bo[1], (long long)bo[2], ckpt);
            const int64_t fsh[3] = {ffF[0], ffF[1], ffF[2]};
            for (int k = 0; k < 10; k++) {
                char d[1400]; snprintf(d, sizeof d, "%s/%s", ffz, ffz_names[k]);
                const size_t held = (size_t)(ffF[0] < FZ ? ffF[0] : FZ) * ffF[1] * ffF[2];
                ffzw[k] = z3w_create(d, fsh, FFS, 0, ffz_fill[k], at);
                ffsl[k] = malloc(held);
                if (!ffzw[k] || !ffsl[k]) { fprintf(stderr, "predict: --fit-zarr: %s\n", z3w_error()); return 1; }
                memset(ffsl[k], ffz_fill[k], held);
            }
            /* a window's cells in z chunks of ffzc planes: 58 device bytes per cell (12 fp32 sums + 10 uint8 fields), so at --fit-h 1 a
               whole 512^3 interior took 7.8 GB and pushed 640^3 windows past the 16 GB card (paged); <= ~0.5 GB now */
            ffwc = stride / ffh + 1;
            { const size_t pc = (size_t)ffwc * ffwc, nzc = (size_t)(strd[0] / ffh + 1), cap = ((size_t)512 << 20) / (58 * pc);
              ffzc = (int)(nzc < (cap ? cap : 1) ? nzc : (cap ? cap : 1)); ffwn = (size_t)ffzc * pc; }
            ffacc = nn_malloc(12 * ffwn * 4); ff8d = nn_malloc(10 * ffwn); ff8h = nn_host_alloc(10 * ffwn);   /* finalized on the device: 10 B per cell back */
            if (!ffacc || !ff8d || !ff8h) { fprintf(stderr, "predict: no memory for the fit-field window accumulator\n"); return 1; }
        }
    }
    if (kss > 1) fprintf(stderr, "predict: one window per %d^3 shards (interior %d)\n", kss, stride);
    double t0 = now(); long ntiles = 0, nskip = 0;
    /* enumerate the tiles in shard order, then let the reader run ahead */
    reader_t rd = {0}; rd.ct = ct; rd.bo = bo; rd.W = W; rd.Wz = Wz; rd.nthreads = nthreads;
    long cap = 0;
    if (kss > 1) for (int64_t sz = 0; sz < nss[0]; sz++) for (int64_t sy = 0; sy < nss[1]; sy++) for (int64_t sx = 0; sx < nss[2]; sx++) {   /* super-shards */
        if (nwork > 1 && (((sz * nss[1] + sy) * nss[2] + sx) % ngpu) != part) continue;
        if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
        rd.tiles[rd.ntiles++] = (tile_t){sz * stride - halo, sy * stride - halo, sx * stride - halo, (sz * nss[1] + sy) * nss[2] + sx};
    }
    if (kss == 1) for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        int64_t so[3] = {sz * shard, sy * shard, sx * shard}, se[3], first[3];
        if (nwork > 1 && (((sz * ns[1] + sy) * ns[2] + sx) % ngpu) != part) continue;   /* another worker's shard */
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
            if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
            rd.tiles[rd.ntiles++] = (tile_t){tz, ty, tx, (sz * ns[1] + sy) * ns[2] + sx};
        }
    }
    for (int k = 0; k < NSLOT; k++) rd.buf[k] = gpath ? nn_host_alloc(w3) : malloc(w3);
    rd.nostats = gpath;
    atomic_store(&rd.filled, 0); atomic_store(&rd.consumed, 0); atomic_store(&rd.failed, 0);
    pthread_mutex_init(&rd.mu, nullptr); pthread_cond_init(&rd.cv, nullptr);
    pthread_t rth; pthread_create(&rth, nullptr, reader_main, &rd);
    int64_t cur_shard = -1, so[3] = {0, 0, 0}, se[3] = {0, 0, 0}, sz = 0, sy = 0, sx = 0;
    int any = 0;
    const int pprof = ufsm_env_on("UFSM_PRED_PROF");
    double p_wait = 0, p_h2d = 0, p_gpu = 0, p_copy = 0, p_wr = 0, pt;
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
        if (kss == 1 && t->shard != cur_shard) {   /* new shard: flush the previous one */
            if (cur_shard >= 0) {
                if (sheet_task && sheet_write_shard(out,sheet_shard,shard,sz,sy,sx)) return 1;
                pt = now();
                if (any) { if (gpath) nn_d2h(sbuf, dsh, WR.bytes); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb ^= 1; sbuf = WR.buf[sb]; }
                p_wr += now() - pt;
                fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   ", (long long)(cur_shard + 1), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
                if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
            }
            cur_shard = t->shard; sz = cur_shard / (ns[1] * ns[2]); sy = (cur_shard / ns[2]) % ns[1]; sx = cur_shard % ns[2];
            so[0] = sz * shard; so[1] = sy * shard; so[2] = sx * shard;
            for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
            if (gpath) nn_zero(dsh, WR.bytes); else memset(sbuf, 0, (size_t)shard * shard * shard);
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
                nn_h2d(ctd, ctu, w3);
                nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
                pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu);
                p_h2d += now() - pt;
            }
            if (nz == 0) { nskip++; pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu); continue; }
            double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
            if (pcut >= 0 && pap_cond == 1) unet_set_cond(u, (float)((pcut - mean) / sd));   /* papyrus: the cutoff in this window's input units */
            if (pcut >= 0 && pap_cond == 2) mean = pcut;                                         /* (or the input's zero) */
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
            if (sheet_task) {
                for (int z=0;z<W;z++) { double a[4]; sheet_parameters(sheet,o[0]+z,a); sheet_rows[4*z]=(float)(o[1]-a[0]); sheet_rows[4*z+1]=(float)(o[2]-a[1]); sheet_rows[4*z+2]=(float)(1/a[2]); sheet_rows[4*z+3]=(float)a[3]; }
                nn_sheet_input(xd,W,sheet_rows,(float)sheet->center,(float)sheet->scale,h16);
            }
            const float *lg;
            if (half) {   /* the network on blocks of the window: its input from the 4-channel window, its logits back to full resolution */
                nn_lowres_input(xd, xs, hdiv, hmode, lrt, x2, h16);
                const float *l2 = unet_forward_x(u, x2, xn, 0, h16);
                nn_lowres_output(l2, unet_logits_h16(u), (shape5){1, 1, W / hdiv, W / hdiv, W / hdiv}, hdiv, hmode, (float *)lrt, lgf);
                lg = lgf;
            } else lg = ct_stem ? unet_forward_ct(u, ctd, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0 ? (int)radial_sign : 0, xs, h16)
                                : unet_forward_x(u, xd, xs, 0, h16);
            nn_set_logits_h16(!half && unet_logits_h16(u));   /* the readers of lg below */
            { static int memi = -1; if (memi < 0) memi = ufsm_env_on("UFSM_PRED_MEMINFO"); if (memi == 1) { nn_sync(); fprintf(stderr, "\npredict: GPU free after the first window %.2f GB\n", nn_mem_free() / 1e9); memi = 2; } }
            if (gn_save) unet_gn_accumulate(u);
            if (ffd && !ffzm) nn_fit_accum_dm(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)tz, (int)ty, (int)tx, ffbox, ffh, ffF, 0, 2, 4, 10, ffds, ffdc, ffacc);
            if (ffzm) {   /* the window interior's whole cells, finalized into the field slab (flushed when the slab is left) */
                int c0[3], c1[3], nc3[3], ok = 1; const int64_t tw[3] = {tz, ty, tx};
                for (int d = 0; d < 3; d++) {
                    int64_t lo = tw[d] + hal[d], hi = tw[d] + wn[d] - hal[d]; if (lo < 0) lo = 0; if (hi > bn[d]) hi = bn[d];
                    c0[d] = (int)((lo + ffh - 1) / ffh); c1[d] = (int)(hi / ffh); if (hi == bn[d]) c1[d] = ffF[d];   /* the box's last, possibly partial, cell */
                    nc3[d] = c1[d] - c0[d]; if (nc3[d] <= 0 || nc3[d] > (d ? ffwc : strd[0] / ffh + 1)) ok = 0;
                }
                if (ok) {
                    const int64_t slab = c0[0] / FZ;
                    while (ffslab < slab) { if (ffz_flush(ffzw, ffsl, ffslab, ffF, FFS, FZ)) { fprintf(stderr, "predict: %s\n", z3w_error()); return 1; } ffslab++; }
                    for (int za = 0; za < nc3[0]; za += ffzc) {   /* z chunks of the window's cells (the accumulator holds ffzc planes) */
                        const int nz = nc3[0] - za < ffzc ? nc3[0] - za : ffzc, ncz[3] = {nz, nc3[1], nc3[2]};
                        const int64_t cz = (int64_t)(c0[0] + za);
                        const size_t nl = (size_t)nz * nc3[1] * nc3[2];
                        nn_zero(ffacc, 12 * nl * 4);
                        const int lbox[3] = {(int)(bn[0] - cz * ffh < (int64_t)nz * ffh ? bn[0] - cz * ffh : (int64_t)nz * ffh),
                                             (int)(bn[1] - (int64_t)c0[1] * ffh < (int64_t)nc3[1] * ffh ? bn[1] - (int64_t)c0[1] * ffh : (int64_t)nc3[1] * ffh),
                                             (int)(bn[2] - (int64_t)c0[2] * ffh < (int64_t)nc3[2] * ffh ? bn[2] - (int64_t)c0[2] * ffh : (int64_t)nc3[2] * ffh)};
                        nn_fit_accum_dm(lg, ctd, ct_shared ? ctm : nullptr, Wz, W, hz, halo, (int)(tz - cz * ffh), (int)(ty - (int64_t)c0[1] * ffh), (int)(tx - (int64_t)c0[2] * ffh), lbox, ffh, ncz, 0, 2, 4, 10, ffds, ffdc, ffacc);
                        nn_fit_finalize(ffacc, nl, ffh, ff8d); nn_d2h(ff8h, ff8d, 10 * nl);
                        for (int a = 0; a < nz; a++) for (int b = 0; b < nc3[1]; b++) for (int k = 0; k < 10; k++)
                            memcpy(ffsl[k] + ((size_t)(cz + a - slab * FZ) * ffF[1] + c0[1] + b) * ffF[2] + c0[2], ff8h + k * nl + ((size_t)a * nc3[1] + b) * nc3[2], (size_t)nc3[2]);
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
            if (gpath && kss > 1) {   /* every shard of this super-shard from the one window, written as it is placed */
                const int64_t SZ = t->shard / (nss[1] * nss[2]), SY = (t->shard / nss[2]) % nss[1], SX = t->shard % nss[2];
                for (int64_t qz = SZ * kss; qz < SZ * kss + kss && qz < ns[0]; qz++) for (int64_t qy = SY * kss; qy < SY * kss + kss && qy < ns[1]; qy++)
                for (int64_t qx = SX * kss; qx < SX * kss + kss && qx < ns[2]; qx++) {
                    const int64_t qo[3] = {qz * shard, qy * shard, qx * shard}, qe[3] = {qo[0] + shard < bn[0] ? qo[0] + shard : bn[0], qo[1] + shard < bn[1] ? qo[1] + shard : bn[1], qo[2] + shard < bn[2] ? qo[2] + shard : bn[2]};
                    nn_zero(dsh, WR.bytes);
                    if (ct_shared) nn_pred_place_m(lg, ctm, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    else nn_pred_place(lg, ctd, W, halo, (int)(tz - qo[0]), (int)(ty - qo[1]), (int)(tx - qo[2]), (int)(qe[0] - qo[0]), (int)(qe[1] - qo[1]), (int)(qe[2] - qo[2]), shard, dsh);
                    nn_d2h(sbuf, dsh, WR.bytes);
                    if (writer_submit(&WR, sb, qz, qy, qx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
                    sb ^= 1; sbuf = WR.buf[sb];
                }
                const char *e = nn_check();
                if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
                if (pprof) { nn_sync(); p_gpu += now() - pt; }
                ntiles++;
                fprintf(stderr, "\rsuper-shard %ld/%ld  %ld tiles (%ld air)  %.0fs   ", i + 1, rd.ntiles, ntiles, nskip, now() - t0);
                continue;
            }
            if (gpath) {   /* interior straight into the device shard buffer */
                if (ct_shared) nn_pred_place_m_d(lg, ctm, Wz, W, hz, halo, (int)(tz - so[0]), (int)(ty - so[1]), (int)(tx - so[2]), (int)(se[0] - so[0]), (int)(se[1] - so[1]), (int)(se[2] - so[2]), shard, dsh);
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
    if (pprof) fprintf(stderr, "\npredict profile (s): main: wait-reader %.2f h2d %.2f gpu(fwd+d2h) %.2f interior-copy %.2f writer-wait %.2f | reader: read %.2f stats %.2f slot-wait %.2f | wall %.2f, %ld tiles\n",
                       p_wait, p_h2d, p_gpu, p_copy, p_wr, rd.t_read, rd.t_stats, rd.t_wait, now() - t0, ntiles);
    if (kss == 1 && cur_shard >= 0 && any) { if (gpath) nn_d2h(sbuf, dsh, WR.bytes); if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } }
    if (sheet_task && cur_shard>=0 && sheet_write_shard(out,sheet_shard,shard,sz,sy,sx)) return 1;
    if (ufsm_env_on("UFSM_PROF")) { fprintf(stderr, "\n"); unet_prof_report(); }
    if (writer_finish(&WR, wth)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    if (ffzm) {   /* the last field slabs, the axis table and fields.json (format zarr) */
        for (; ffslab * FZ < ffF[0]; ffslab++) if (ffz_flush(ffzw, ffsl, ffslab, ffF, FFS, FZ)) { fprintf(stderr, "predict: %s\n", z3w_error()); return 1; }
        for (int k = 0; k < 10; k++) { z3w_close(ffzw[k]); free(ffsl[k]); }
        nn_free(ffacc); nn_free(ff8d); nn_host_free(ff8h);
        char p[1400]; snprintf(p, sizeof p, "%s/fields.json", ffz); FILE *f = fopen(p, "w");
        if (!f) { fprintf(stderr, "predict: cannot write %s\n", p); return 1; }
        fprintf(f, "{\"format\": \"zarr\", \"origin\": [%lld, %lld, %lld], \"shape\": [%d, %d, %d], \"h\": %d, \"um\": %.9g, \"axis\": [", (long long)bo[0], (long long)bo[1], (long long)bo[2], ffF[0], ffF[1], ffF[2], ffh, um * (1 << level));
        for (int64_t z = 0; z <= bn[0]; z++) { double cy = 0, cx = 0; if (ax.n) axis_at(&ax, (double)(bo[0] + z) * (1 << level), &cy, &cx); fprintf(f, "%s[%.3f, %.3f]", z ? ", " : "", cy / (1 << level) - bo[1], cx / (1 << level) - bo[2]); }
        fprintf(f, "]}\n"); fclose(f);
        fprintf(stderr, "predict: fit fields %dx%dx%d (h %d) written to %s (zarr)\n", ffF[0], ffF[1], ffF[2], ffh, ffz);
    } else if (ffd) {   /* the fit fields: means over material (CT > 0) of the pooled heads, material fraction, max recto */
        float *acc = malloc(12 * ffn * 4); nn_sync(); nn_d2h(acc, ffacc, 12 * ffn * 4); nn_free(ffacc);
        uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(ffn, 1);
        fit_finalize(acc, ffn, ffh, a);
        const int na = (int)bn[0] + 1; float *cyb = malloc(na * sizeof(float)), *cxb = malloc(na * sizeof(float));
        for (int z = 0; z < na; z++) { double cy = 0, cx = 0; if (ax.n) axis_at(&ax, (double)(bo[0] + z) * (1 << level), &cy, &cx); cyb[z] = (float)(cy / (1 << level) - bo[1]); cxb[z] = (float)(cx / (1 << level) - bo[2]); }
        const long long org[3] = {bo[0], bo[1], bo[2]};
        if (fit_fields_save(ffd, org, ffF, ffh, um * (1 << level), a, cyb, cxb, na)) { fprintf(stderr, "predict: cannot write the fit fields to %s\n", ffd); return 1; }
        fprintf(stderr, "predict: fit fields %dx%dx%d (h %d) written to %s\n", ffF[0], ffF[1], ffF[2], ffh, ffd);
        for (int k = 0; k < 10; k++) free(a[k]);
        free(acc); free(cyb); free(cxb);
    }
    if (gpath) { nn_host_free(WR.buf[0]); nn_host_free(WR.buf[1]); nn_free(dsh); } else { free(WR.buf[0]); free(WR.buf[1]); }
    nn_free(stsc); nn_free(x2); nn_free(lgf); nn_free(lrt);
    pthread_join(rth, nullptr);
    for (int k = 0; k < NSLOT; k++) { if (gpath) nn_host_free(rd.buf[k]); else free(rd.buf[k]); }
    free(rd.tiles);
    fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   \n", (long long)(ns[0] * ns[1] * ns[2]), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
    z3w_close(w);
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
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
    if (pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs)) return 1;
    fprintf(stderr, "wrote %s (%ld tiles in %.0fs)\n", out, ntiles, now() - t0);
    z3_close(ct); store_close(s); unet_free(u);
    free(sheet_window); free(sheet_shard); free(sheet_rows); sheet_free(sheet);
    return 0;
}
