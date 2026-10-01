/* ufsm predict: sliding-window inference of a checkpoint over a CT box, written as a volcomp zarr v3
   probability pyramid (uint8 p*255, q=8) in the published layout. */
#include "nn.h"
#include "pyramid.h"
#include "sources.h"
#include "store.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
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
    z3 *ct; const int64_t *bo; int W, nthreads;
    tile_t *tiles; long ntiles;
    uint8_t *buf[NSLOT]; size_t nz[NSLOT]; double sum[NSLOT], sq[NSLOT];
    atomic_long filled;          /* tiles read so far (slot = i % NSLOT) */
    atomic_long consumed;        /* tiles released by the main thread */
    atomic_int failed;
    pthread_mutex_t mu; pthread_cond_t cv;
} reader_t;
static void *reader_main(void *arg) {
    reader_t *r = arg;
    size_t w3 = (size_t)r->W * r->W * r->W;
    for (long i = 0; i < r->ntiles && !atomic_load(&r->failed); i++) {
        pthread_mutex_lock(&r->mu);
        while (i - atomic_load(&r->consumed) >= NSLOT) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        int k = (int)(i % NSLOT);
        tile_t *t = &r->tiles[i];
        int64_t o[3] = {r->bo[0] + t->tz, r->bo[1] + t->ty, r->bo[2] + t->tx}, n[3] = {r->W, r->W, r->W};
        if (z3_read(r->ct, o, n, r->buf[k], r->nthreads)) { atomic_store(&r->failed, 1); }
        else {
            const uint8_t *c = r->buf[k]; size_t nz = 0; double sum = 0, sq = 0;
            for (size_t j = 0; j < w3; j++) { nz += c[j] != 0; sum += c[j]; sq += (double)c[j] * c[j]; }
            r->nz[k] = nz; r->sum[k] = sum; r->sq[k] = sq;
        }
        pthread_mutex_lock(&r->mu); atomic_store(&r->filled, i + 1); pthread_cond_broadcast(&r->cv); pthread_mutex_unlock(&r->mu);
    }
    pthread_mutex_lock(&r->mu); pthread_cond_broadcast(&r->cv); pthread_mutex_unlock(&r->mu);
    return nullptr;
}
/* shard writer: the main thread hands a filled shard buffer to a writer thread and continues into the other buffer */
typedef struct { z3w *w; int nthreads; size_t bytes; uint8_t *buf[2]; int64_t sz[2], sy[2], sx[2]; int pending[2]; int failed; pthread_mutex_t mu; pthread_cond_t cv; int stop; } writer_t;
static void *writer_main(void *arg) {
    writer_t *W = arg;
    for (;;) {
        pthread_mutex_lock(&W->mu);
        int k = -1;
        while (k < 0 && !W->stop) { for (int i = 0; i < 2; i++) if (W->pending[i]) { k = i; break; } if (k < 0) pthread_cond_wait(&W->cv, &W->mu); }
        if (k < 0) { pthread_mutex_unlock(&W->mu); return nullptr; }
        pthread_mutex_unlock(&W->mu);
        if (z3w_write_shard(W->w, W->sz[k], W->sy[k], W->sx[k], W->buf[k], W->nthreads)) W->failed = 1;
        pthread_mutex_lock(&W->mu); W->pending[k] = 0; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
}
/* queue buffer k for writing; returns when the OTHER buffer is free to fill */
static int writer_submit(writer_t *W, int k, int64_t sz, int64_t sy, int64_t sx) {
    pthread_mutex_lock(&W->mu);
    W->sz[k] = sz; W->sy[k] = sy; W->sx[k] = sx; W->pending[k] = 1; pthread_cond_broadcast(&W->cv);
    while (W->pending[k ^ 1]) pthread_cond_wait(&W->cv, &W->mu);
    pthread_mutex_unlock(&W->mu);
    return W->failed;
}
static int writer_finish(writer_t *W, pthread_t th) {
    pthread_mutex_lock(&W->mu); while (W->pending[0] || W->pending[1]) pthread_cond_wait(&W->cv, &W->mu); W->stop = 1; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    pthread_join(th, nullptr); return W->failed;
}
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

int cmd_predict(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm predict <ckpt> <root> <ct-group-key> <out-dir> --um U [--level 0] [--box z,y,x,nz,ny,nx] [--window 288] [--act-mx8 1]\n"
                        "       [--halo 16] [--shard 512] [--gpu 0] [--cache DIR] [--axis umbilicus.json] [--levels 4] [--q 8] [--threads 16]\n"
                        "       [--prec 1|2|3|4] [--policy enc0=1,...]   inference precision (fp8 / fp4 compute)\n"
                        "       [--gpus 0,1]   one worker per GPU over the shards of the same output (needs --box)\n");
        fprintf(stderr, "  window - 2*halo should divide the shard size (288 - 32 = 256 divides 256 and 512): tiles then cover each shard exactly. 288^3 needs ~5.5 GB; 160 for small GPUs.\n");
        return 2;
    }
    const char *ckpt = argv[2], *root = argv[3], *key = argv[4], *out = argv[5];
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), W = atoi(opt(argc, argv, "--window", "288")), halo = atoi(opt(argc, argv, "--halo", "8"));
    int shard = atoi(opt(argc, argv, "--shard", "512")), gpu = atoi(opt(argc, argv, "--gpu", "0")), nlev = atoi(opt(argc, argv, "--levels", "4"));
    int nthreads = atoi(opt(argc, argv, "--threads", "16"));
    float q = (float)atof(opt(argc, argv, "--q", "8"));
    const char *cache = opt(argc, argv, "--cache", nullptr), *axisf = opt(argc, argv, "--axis", nullptr);
    nn_set_prec(atoi(opt(argc, argv, "--prec", "1")));
    if (nn_set_prec_policy(opt(argc, argv, "--policy", ""))) return 2;
    if (atoi(opt(argc, argv, "--fp4", "0"))) { unet_set_act_mx4(1); if (!*opt(argc, argv, "--prec", "")) nn_set_prec(3); }   /* fp4 storage + fp4 compute */
    if (atoi(opt(argc, argv, "--act-mx4", "0"))) unet_set_act_mx4(1);   /* packed fp4 activation storage (default once yardstick A passes) */
    else if (atoi(opt(argc, argv, "--act-mx8", "1"))) unet_set_act_mx8(1);   /* default on: free in accuracy (r5 0.197 vs 0.192, r8 0.2670 vs 0.2669), 1.3-1.5x faster */   /* MX-fp8 activation storage for this (inference-only) process */
    if (um <= 0) { fprintf(stderr, "--um required\n"); return 2; }
    /* --gpus a,b,...: one worker process per GPU, each writes every n-th shard of the same store; the parent builds the pyramid */
    int gpus[8], ngpu = 0, part = 0;
    { const char *gl = opt(argc, argv, "--gpus", nullptr); if (gl) { char *t = strdup(gl); for (char *q = strtok(t, ","); q && ngpu < 8; q = strtok(nullptr, ",")) gpus[ngpu++] = atoi(q); free(t); } }
    if (ngpu <= 1) { if (ngpu == 1) gpu = gpus[0]; ngpu = 1; }
    pid_t kids[8] = {0}; int is_child = 0;
    if (ngpu > 1) {
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
            char attrs[2000]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"recto probability (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"gpus\":%d}}", ckpt, ngpu);
            for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
            pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs);
            fprintf(stderr, "wrote %s (%d GPUs)\n", out, ngpu);
            return 0;
        }
    }
    if (nn_init(gpu)) { fprintf(stderr, "cannot select GPU %d\n", gpu); return 1; }
    unet_cfg cfg; int step;
    if (unet_peek(ckpt, &cfg, &step)) { fprintf(stderr, "cannot read %s\n", ckpt); return 1; }
    unet *u = unet_create(&cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "cannot load %s\n", ckpt); return 1; }
    unet_use_ema(u, 1);
    if (W % (1 << (cfg.nlev - 1))) { fprintf(stderr, "window must be divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
    store *s = store_open(root);
    char ak[1200];
    snprintf(ak, sizeof ak, "%s/level%d", key, level);
    z3 *ct = pyramid_open_level(s, key, level, um, cache);
    if (!ct) { fprintf(stderr, "cannot open level %d of %s: %s\n", level, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    { const char *b = opt(argc, argv, "--box", nullptr); long long v[6]; if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } } }
    axis ax = {0};
    if (axisf && axis_load(&ax, axisf)) { fprintf(stderr, "cannot load axis %s\n", axisf); return 1; }
    double um_l = um * (1 << level), scale = (double)(1 << level);
    /* output array covers the box, with origin recorded in attributes */
    char lv[32], ldir[1400], attrs[1024];
    z3w_level_name(um_l, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"recto probability (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"step\":%d,\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d}}",
             ckpt, step, root, ak, (long long)bo[0], (long long)bo[1], (long long)bo[2], W, halo);
    z3w *w = z3w_create(ldir, bn, shard, q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    int stride = W - 2 * halo;
    size_t w3 = (size_t)W * W * W;
    writer_t WR = {0}; WR.w = w; WR.nthreads = nthreads; WR.bytes = (size_t)shard * shard * shard;
    WR.buf[0] = malloc(WR.bytes); WR.buf[1] = malloc(WR.bytes); pthread_mutex_init(&WR.mu, nullptr); pthread_cond_init(&WR.cv, nullptr);
    pthread_t wth; pthread_create(&wth, nullptr, writer_main, &WR);
    int sb = 0; uint8_t *sbuf = WR.buf[0];
    /* the window goes up as uint8; the input channels are built on the device in the network's storage type */
    const int h16 = nn_get_tf32() && nn_get_act_bf16() && !getenv("UFSM_ACT_MX8");
    uint8_t *ctd = nn_malloc(w3), *pu = malloc(w3), *pud = nn_malloc(w3);
    float *dyo = malloc(2 * (size_t)W * sizeof(float)), *dxo = dyo + W, *dyd = nn_malloc(2 * (size_t)W * sizeof(float)), *dxd = dyd + W;
    void *xd = nn_malloc(4 * w3 * (h16 ? 2 : 4));
    shape5 xs = {1, 4, W, W, W};
    int64_t ns[3];
    for (int d = 0; d < 3; d++) ns[d] = (bn[d] + shard - 1) / shard;
    double t0 = now(); long ntiles = 0, nskip = 0;
    /* enumerate the tiles in shard order, then let the reader run ahead */
    reader_t rd = {0}; rd.ct = ct; rd.bo = bo; rd.W = W; rd.nthreads = nthreads;
    long cap = 0;
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        int64_t so[3] = {sz * shard, sy * shard, sx * shard}, se[3];
        if (ngpu > 1 && (((sz * ns[1] + sy) * ns[2] + sx) % ngpu) != part) continue;   /* another worker's shard */
        for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
        for (int64_t tz = so[0] - halo; tz + halo < se[0]; tz += stride)
        for (int64_t ty = so[1] - halo; ty + halo < se[1]; ty += stride)
        for (int64_t tx = so[2] - halo; tx + halo < se[2]; tx += stride) {
            if (rd.ntiles == cap) { cap = cap ? 2 * cap : 1024; rd.tiles = realloc(rd.tiles, (size_t)cap * sizeof *rd.tiles); }
            rd.tiles[rd.ntiles++] = (tile_t){tz, ty, tx, (sz * ns[1] + sy) * ns[2] + sx};
        }
    }
    for (int k = 0; k < NSLOT; k++) rd.buf[k] = malloc(w3);
    atomic_store(&rd.filled, 0); atomic_store(&rd.consumed, 0); atomic_store(&rd.failed, 0);
    pthread_mutex_init(&rd.mu, nullptr); pthread_cond_init(&rd.cv, nullptr);
    pthread_t rth; pthread_create(&rth, nullptr, reader_main, &rd);
    int64_t cur_shard = -1, so[3] = {0, 0, 0}, se[3] = {0, 0, 0}, sz = 0, sy = 0, sx = 0;
    int any = 0;
    for (long i = 0; i < rd.ntiles; i++) {
        pthread_mutex_lock(&rd.mu);
        while (atomic_load(&rd.filled) <= i && !atomic_load(&rd.failed)) pthread_cond_wait(&rd.cv, &rd.mu);
        pthread_mutex_unlock(&rd.mu);
        if (atomic_load(&rd.failed)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const tile_t *t = &rd.tiles[i];
        if (t->shard != cur_shard) {   /* new shard: flush the previous one */
            if (cur_shard >= 0) {
                if (any) { if (writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; } sb ^= 1; sbuf = WR.buf[sb]; }
                fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   ", (long long)(cur_shard + 1), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
            }
            cur_shard = t->shard; sz = cur_shard / (ns[1] * ns[2]); sy = (cur_shard / ns[2]) % ns[1]; sx = cur_shard % ns[2];
            so[0] = sz * shard; so[1] = sy * shard; so[2] = sx * shard;
            for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
            memset(sbuf, 0, (size_t)shard * shard * shard); any = 0;
        }
        int k = (int)(i % NSLOT);
        const uint8_t *ctu = rd.buf[k];
        const int64_t tz = t->tz, ty = t->ty, tx = t->tx, o[3] = {bo[0] + tz, bo[1] + ty, bo[2] + tx};
        {
            size_t nz = rd.nz[k]; double sum = rd.sum[k], sq = rd.sq[k];
            if (nz == 0) { nskip++; pthread_mutex_lock(&rd.mu); atomic_store(&rd.consumed, i + 1); pthread_cond_broadcast(&rd.cv); pthread_mutex_unlock(&rd.mu); continue; }
            double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
            for (int z = 0; z < W; z++) {   /* window origin relative to the axis at this slice (double on the host, small in float) */
                double cy, cx;
                axis_at(&ax, (double)(o[0] + z) * scale, &cy, &cx);
                cy /= scale; cx /= scale;
                dyo[z] = (float)((double)o[1] - cy); dxo[z] = (float)((double)o[2] - cx);
            }
            nn_h2d(ctd, ctu, w3); nn_h2d(dyd, dyo, 2 * (size_t)W * sizeof(float));
            nn_pred_input(ctd, W, (float)mean, (float)(1.0 / sd), dyd, dxd, ax.n > 0, xd, h16);
            const float *lg = unet_forward_x(u, xd, xs, 0, h16);
            nn_pred_output(lg, ctd, w3, pud);      /* channel 0 = recto */
            nn_d2h(pu, pud, w3);
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
        }
    }
    if (cur_shard >= 0 && any && writer_submit(&WR, sb, sz, sy, sx)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    if (writer_finish(&WR, wth)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    free(WR.buf[0]); free(WR.buf[1]);
    pthread_join(rth, nullptr);
    for (int k = 0; k < NSLOT; k++) free(rd.buf[k]);
    free(rd.tiles);
    fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   \n", (long long)(ns[0] * ns[1] * ns[2]), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
    z3w_close(w);
    if (ngpu > 1) { fprintf(stderr, "gpu %d: %ld tiles in %.0fs\n", gpu, ntiles, now() - t0); z3_close(ct); store_close(s); unet_free(u); return 0; }   /* worker: the parent builds the pyramid */
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
    pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs);
    fprintf(stderr, "wrote %s (%ld tiles in %.0fs)\n", out, ntiles, now() - t0);
    z3_close(ct); store_close(s); unet_free(u);
    return 0;
}
