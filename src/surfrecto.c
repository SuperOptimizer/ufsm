/* ufsm surfrecto-predict: the published recto-surface U-Net (src/surfrecto.h) over a box of a CT zarr (2.4 um level 0), following
   villa vesuvius.models.run.inference / blending / finalize_outputs:
   - windows of P (256) along each axis: a grid of stride S = P (1 - overlap) (overlap 0.5 -> 128) centred on the box widened by
     (P - S) / 2 per side (so every box voxel lies in the central S of some window), starts clamped into the volume; one window row
     (P z, P y, all x) of CT is read at a time (prefetched by a reader thread);
   - per window: z-score over the window (mean, population std of the uint8 values; villa instance_zscore); windows whose CT is
     constant (min == max, e.g. masked air) are skipped as villa's is_empty, also windows whose max is below --air when given;
   - optional mirror TTA (--tta 1: the 8 flips, mean of the logit margins);
   - blend: sum g m / sum g with g = villa generate_gaussian_map (sigma P / 8, peak 1 at P / 2), m = l1 - l0; p = sigmoid (2-class
     softmax), uint8 = rint(255 p); voxels without any window weight = 0 (villa writes uncovered voxels as background).
   Output: <out>/<um> (OME group <out>), uint8, the FULL level-0 shape of the CT array (a voxel (z, y, x) of the output is the CT
   voxel (z, y, x)), 512^3 shards; only shards touching the box are written, everything outside the box is 0 (fill). Accumulation
   runs on the host per output shard (fp32 sum g m and sum g over the shard's part of the box), a shard is finalised and written as
   soon as the last window touching it is blended (blend + write on a second thread, overlapped with the GPU). */
#include "surfrecto.h"
#include "sysinfo.h"
#include "store.h"
#include "xnet.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[1400]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "surfrecto: cannot write %s\n", path); return -1; }
    const size_t w = fwrite(p, 1, n, f); fclose(f); return w == n ? 0 : -1;
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

/* one window through the network (+ TTA); margin on the device (net-owned or tta buffer) */
static const float *window_fwd(sr_net *net, const uint8_t *wd, uint8_t *fl, float *tacc, int P, int tta, float mean, float sd) {
    if (!tta) return sr_forward(net, wd, P, mean, sd);
    for (int m = 0; m < 8; m++) {
        if (m) sr_flip_u8(wd, fl, P, m);
        const float *o = sr_forward(net, m ? fl : wd, P, mean, sd); if (!o) return nullptr;
        sr_flip_acc(tacc, o, P, m, m == 0);
    }
    sr_scale(tacc, (size_t)P * P * P, 1.f / 8.f);
    return tacc;
}

/* --raw-in: one window (parity with the PyTorch reference) */
static int raw_mode(sr_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr);
    const int P = atoi(opt(argc, argv, "--window", "256")), reps = atoi(opt(argc, argv, "--reps", "1")), keep = atoi(opt(argc, argv, "--keep", "0"));
    const int tta = atoi(opt(argc, argv, "--tta", "0"));
    if (!in || !out) { fprintf(stderr, "surfrecto-predict --ckpt PTH --raw-in WINDOW.u8 --raw-out DIR [--window 256] [--keep 1] [--tta 1] [--reps N]\n"); return 2; }
    const size_t nv = (size_t)P * P * P; uint8_t *u = malloc(nv);
    FILE *f = fopen(in, "rb"); if (!f || !u || fread(u, 1, nv, f) != nv) { fprintf(stderr, "surfrecto: %s is not %d^3 bytes\n", in, P); return 1; } fclose(f);
    double mean, sd; int vmin, vmax; sr_zstats(u, nv, &mean, &sd, &vmin, &vmax);
    fprintf(stderr, "surfrecto: window stats mean %.6f std %.6f min %d max %d\n", mean, sd, vmin, vmax);
    uint8_t *ud = xmalloc_dev(nv), *fl = tta ? xmalloc_dev(nv) : nullptr; float *tacc = tta ? xmalloc_dev(nv * 4) : nullptr;
    xh2d(ud, u, nv);
    sr_keep(net, keep);
    const float *m = nullptr; double best = 1e30;
    for (int r = 0; r < reps; r++) {
        xnet_sync(); const double t0 = now_s();
        m = window_fwd(net, ud, fl, tacc, P, tta, (float)mean, (float)sd); if (!m) return 1;
        xnet_sync(); const double t = now_s() - t0; if (t < best) best = t;
    }
    fprintf(stderr, "surfrecto: window %d^3 forward%s %.1f ms (best of %d) = %.2f Mvox/s, device peak cache %.2f GB\n", P, tta ? " (8-flip TTA)" : "", best * 1e3, reps,
            (double)nv / best / 1e6, (double)(xnet_mem_used() + xnet_mem_cached()) / 1e9);
    { char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1; }
    float *mh = malloc(nv * 4); xd2h(mh, m, nv * 4);
    if (write_file(out, "margin.f32", mh, nv * 4)) return 1;
    free(mh);
    for (int k = 0; keep && k < 13; k++) {
        int e, ch; const void *p = sr_kept(net, k, &e, &ch); if (!p) continue;
        const size_t n = (size_t)e * e * e * ch; uint16_t *h = malloc(n * 2); float *fo = malloc(n * 4);
        xd2h(h, p, n * 2); for (size_t i = 0; i < n; i++) fo[i] = h2f(h[i]);
        char nm[32]; snprintf(nm, sizeof nm, "%s%d_%d_%d.f32", k < 7 ? "enc" : "dec", k < 7 ? k : k - 7, e, ch);
        if (write_file(out, nm, fo, n * 4)) return 1;
        free(h); free(fo);
    }
    char st[128]; snprintf(st, sizeof st, "%.9g %.9g\n", mean, sd);
    if (write_file(out, "stats.txt", st, strlen(st))) return 1;
    free(u); xfree_dev(ud); xfree_dev(fl); xfree_dev(tacc);
    return 0;
}

/* window starts along one axis: stride S grid centred on [b0 - m, b1 + m), m = (P - S) / 2, clamped into [0, V - P], deduplicated */
typedef struct { int n; int64_t *s; } grid1;
static grid1 grid_of(int64_t b0, int64_t b1, int64_t V, int P, int S) {
    const int64_t m = (P - S) / 2, span = b1 - b0 + 2 * m;
    const int64_t n = span <= P ? 1 : (span - P + S - 1) / S + 1, total = (n - 1) * S + P, c0 = b0 - m - (total - span) / 2;
    grid1 g = {0, malloc(sizeof(int64_t) * (size_t)n)};
    for (int64_t k = 0; k < n; k++) {
        int64_t s = c0 + k * S; if (s > V - P) s = V - P; if (s < 0) s = 0;
        if (!g.n || g.s[g.n - 1] != s) g.s[g.n++] = s;
    }
    return g;
}

typedef struct { int64_t o[3], n[3]; int64_t idx[3]; long long last; float *acc, *w; } shard_t;
typedef struct { long long win; int air; float *m; int iz, iy, ix; } job_t;
enum { NSLOT = 3 };
typedef struct {
    /* static */
    int P, SH, nth; int64_t box0[3], box1[3]; grid1 G[3];
    shard_t *sh; int nsh; float *g1;
    z3w *zw; uint8_t *sbuf;
    /* blend queue */
    pthread_mutex_t mu; pthread_cond_t cv;
    job_t q[NSLOT]; int qh, qt, qn, done;
    float *slots[NSLOT]; int slot_free[NSLOT];
    int err; long long written, live, live_max; double tblend, twrite;
    /* the blend parallel-for */
    shard_t *cur_sh[64]; int ncur; const float *cur_m; int64_t cur_w0[3];
    atomic_int next_task; int ntask;
} ctx_t;

/* parallel over (touched shard, z row) */
typedef struct { ctx_t *C; } bw_arg;
static void blend_rows(ctx_t *C) {
    const int P = C->P;
    for (;;) {
        const int t = atomic_fetch_add(&C->next_task, 1); if (t >= C->ntask) break;
        int k = 0, rem = t;
        for (; k < C->ncur; k++) {
            shard_t *S = C->cur_sh[k];
            const int64_t z0 = S->o[0] > C->cur_w0[0] ? S->o[0] : C->cur_w0[0], z1 = S->o[0] + S->n[0] < C->cur_w0[0] + P ? S->o[0] + S->n[0] : C->cur_w0[0] + P;
            const int nz = (int)(z1 - z0); if (rem < nz) break; rem -= nz;
        }
        if (k >= C->ncur) break;
        shard_t *S = C->cur_sh[k]; int64_t lo[3], hi[3];
        for (int a = 0; a < 3; a++) {
            lo[a] = S->o[a] > C->cur_w0[a] ? S->o[a] : C->cur_w0[a];
            hi[a] = S->o[a] + S->n[a] < C->cur_w0[a] + P ? S->o[a] + S->n[a] : C->cur_w0[a] + P;
        }
        const int64_t z = lo[0] + rem;
        const int wz = (int)(z - C->cur_w0[0]); const float gz = C->g1[wz];
        for (int64_t y = lo[1]; y < hi[1]; y++) {
            const int wy = (int)(y - C->cur_w0[1]); const float gzy = gz * C->g1[wy];
            const float *mr = C->cur_m + ((size_t)wz * P + wy) * P + (lo[2] - C->cur_w0[2]);
            const float *gx = C->g1 + (lo[2] - C->cur_w0[2]);
            const size_t base = ((size_t)(z - S->o[0]) * S->n[1] + (size_t)(y - S->o[1])) * S->n[2] + (size_t)(lo[2] - S->o[2]);
            float *ar = S->acc + base, *wr = S->w + base;
            const int nx = (int)(hi[2] - lo[2]);
            for (int x = 0; x < nx; x++) { const float g = gzy * gx[x]; ar[x] += g * mr[x]; wr[x] += g; }
        }
    }
}
static void *blend_worker(void *p) { blend_rows(((bw_arg *)p)->C); return nullptr; }

static int finalize_shard(ctx_t *C, shard_t *S) {
    if (!S->acc) return 0;   /* no window data: nothing written (fill 0) */
    const double t0 = now_s(); const int SH = C->SH;
    memset(C->sbuf, 0, (size_t)SH * SH * SH);
    const int64_t so[3] = {S->idx[0] * SH, S->idx[1] * SH, S->idx[2] * SH};
    for (int64_t z = 0; z < S->n[0]; z++) for (int64_t y = 0; y < S->n[1]; y++) {
        const size_t src = ((size_t)z * S->n[1] + y) * S->n[2];
        uint8_t *dst = C->sbuf + ((size_t)(S->o[0] - so[0] + z) * SH + (size_t)(S->o[1] - so[1] + y)) * SH + (size_t)(S->o[2] - so[2]);
        for (int64_t x = 0; x < S->n[2]; x++) {
            const float w = S->w[src + x];
            if (!(w > 0.f)) { dst[x] = 0; continue; }
            float l = S->acc[src + x] / w; l = l > 60.f ? 60.f : l < -60.f ? -60.f : l;
            dst[x] = (uint8_t)lrintf(255.f / (1.f + expf(-l)));
        }
    }
    free(S->acc); free(S->w); S->acc = S->w = nullptr; C->live--;
    if (z3w_write_shard(C->zw, S->idx[0], S->idx[1], S->idx[2], C->sbuf, C->nth)) { fprintf(stderr, "surfrecto: %s\n", z3w_error()); return -1; }
    C->written++; C->twrite += now_s() - t0;
    return 0;
}

static void *blend_main(void *p) {
    ctx_t *C = p; const int P = C->P;
    long long fin_from = 0;   /* shards sorted by last: finalise in that order */
    for (;;) {
        pthread_mutex_lock(&C->mu);
        while (!C->qn && !C->done) pthread_cond_wait(&C->cv, &C->mu);
        if (!C->qn) { pthread_mutex_unlock(&C->mu); break; }
        job_t J = C->q[C->qh]; pthread_mutex_unlock(&C->mu);
        if (!J.air && !C->err) {
            const double t0 = now_s();
            const int64_t w0[3] = {C->G[0].s[J.iz], C->G[1].s[J.iy], C->G[2].s[J.ix]};
            C->ncur = 0; C->ntask = 0;
            for (int i = 0; i < C->nsh; i++) {
                shard_t *S = &C->sh[i]; int hit = 1;
                for (int a = 0; a < 3; a++) hit &= S->o[a] < w0[a] + P && S->o[a] + S->n[a] > w0[a];
                if (!hit) continue;
                if (!S->acc) {
                    const size_t n = (size_t)S->n[0] * S->n[1] * S->n[2];
                    S->acc = calloc(n, 4); S->w = calloc(n, 4);
                    if (!S->acc || !S->w) { fprintf(stderr, "surfrecto: out of host memory (shard accumulators)\n"); C->err = 1; break; }
                    if (++C->live > C->live_max) C->live_max = C->live;
                }
                if (C->ncur == 64) { fprintf(stderr, "surfrecto: a window touches more than 64 shards\n"); C->err = 1; break; }
                C->cur_sh[C->ncur++] = S;
                const int64_t z0 = S->o[0] > w0[0] ? S->o[0] : w0[0], z1 = S->o[0] + S->n[0] < w0[0] + P ? S->o[0] + S->n[0] : w0[0] + P;
                C->ntask += (int)(z1 - z0);
            }
            if (!C->err) {
                memcpy(C->cur_w0, w0, sizeof w0); C->cur_m = J.m; atomic_store(&C->next_task, 0);
                pthread_t th[64]; bw_arg A = {C}; int nt = 0;
                for (int t = 1; t < C->nth && t < 64; t++) if (!pthread_create(&th[nt], nullptr, blend_worker, &A)) nt++;
                blend_rows(C);
                for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
            }
            C->tblend += now_s() - t0;
        }
        /* finalise the shards whose last window this was (C->sh sorted by last) */
        while (fin_from < C->nsh && C->sh[fin_from].last <= J.win) { if (!C->err && finalize_shard(C, &C->sh[fin_from])) C->err = 1; fin_from++; }
        pthread_mutex_lock(&C->mu);
        if (!J.air) for (int s = 0; s < NSLOT; s++) if (C->slots[s] == J.m) C->slot_free[s] = 1;
        C->qh = (C->qh + 1) % NSLOT; C->qn--;
        pthread_cond_broadcast(&C->cv); pthread_mutex_unlock(&C->mu);
    }
    return nullptr;
}
static void push_job(ctx_t *C, job_t J) {
    pthread_mutex_lock(&C->mu);
    while (C->qn == NSLOT) pthread_cond_wait(&C->cv, &C->mu);
    C->q[C->qt] = J; C->qt = (C->qt + 1) % NSLOT; C->qn++;
    pthread_cond_broadcast(&C->cv); pthread_mutex_unlock(&C->mu);
}
static float *get_slot(ctx_t *C) {
    pthread_mutex_lock(&C->mu);
    for (;;) {
        for (int s = 0; s < NSLOT; s++) if (C->slot_free[s]) { C->slot_free[s] = 0; pthread_mutex_unlock(&C->mu); return C->slots[s]; }
        pthread_cond_wait(&C->cv, &C->mu);
    }
}
static int cmp_last(const void *a, const void *b) { const long long x = ((const shard_t *)a)->last, y = ((const shard_t *)b)->last; return x < y ? -1 : x > y; }

/* CT row reader (prefetch): row r = (iz, iy) -> buffer [P][P][W] */
typedef struct {
    z3 *ct; int P, nth; grid1 *G; int64_t W; int nrows;
    uint8_t *buf[2]; int ready[2], row_of[2]; int stop, err;
    pthread_mutex_t mu; pthread_cond_t cv; double tread;
} reader_t;
static void *reader_main(void *p) {
    reader_t *R = p;
    for (int r = 0; r < R->nrows; r++) {
        const int b = r & 1;
        pthread_mutex_lock(&R->mu);
        while (R->ready[b] && !R->stop) pthread_cond_wait(&R->cv, &R->mu);
        const int stop = R->stop; pthread_mutex_unlock(&R->mu);
        if (stop) break;
        const int iz = r / R->G[1].n, iy = r % R->G[1].n;
        const int64_t o[3] = {R->G[0].s[iz], R->G[1].s[iy], R->G[2].s[0]}, n[3] = {R->P, R->P, R->W};
        const double t0 = now_s();
        const int rc = z3_read(R->ct, o, n, R->buf[b], R->nth);
        pthread_mutex_lock(&R->mu);
        R->tread += now_s() - t0;
        if (rc) { fprintf(stderr, "surfrecto: CT read: %s\n", z3_error()); R->err = 1; }
        R->ready[b] = 1; R->row_of[b] = r;
        pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
        if (rc) break;
    }
    return nullptr;
}

int cmd_surfrecto_predict(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr);
    const int raw = opt(argc, argv, "--raw-in", nullptr) != nullptr;
    if (!raw && argc >= 6 && argv[2][0] != '-') ck = argv[2];
    if (!ck || (!raw && argc < 6)) {
        fprintf(stderr, "usage: ufsm surfrecto-predict CKPT CT-ROOT CT-KEY OUT --box z0,y0,x0,z1,y1,x1 [--um 2.4] [--window 256] [--overlap 0.5]\n"
                        "         [--tta 0] [--air 0] [--shard 512] [--gpu 0] [--gpu-mem GB] [--threads N] [--cache DIR] [--part i/n]\n"
                        "  scrollprize/surface_recto_3dunet (checkpoint_inference_ready.pth) on a CT box (level-0 voxels, end-exclusive; the model\n"
                        "  works on 2.4 um CT). OUT/<um>: uint8 rint(255 p(recto surface)), full CT shape (voxel = CT voxel), 0 outside the box.\n"
                        "  --tta 1: 8-flip mirror TTA (villa default, 8x the time). --air A: also skip windows whose CT max < A.\n"
                        "  --part i/n: compute only the i-th of n shares of the box's 512-row y shard rows (one process per GPU into the same OUT).\n"
                        "  ufsm surfrecto-predict --ckpt PTH --raw-in WINDOW.u8 --raw-out DIR [--window 256] [--keep 1] [--tta 1]   one window (parity)\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (opt(argc, argv, "--gpu-mem", nullptr)) xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "8")) * 1e9));
    const double tl = now_s();
    sr_net *net = sr_load(ck); if (!net) return 1;
    fprintf(stderr, "surfrecto: loaded %s in %.1f s\n", ck, now_s() - tl);
    if (raw) { const int rc = raw_mode(net, argc, argv); sr_free(net); return rc; }

    const char *root = argv[3], *key = argv[4], *out = argv[5];
    const int P = atoi(opt(argc, argv, "--window", "256")), tta = atoi(opt(argc, argv, "--tta", "0")), air = atoi(opt(argc, argv, "--air", "0"));
    const int SH = atoi(opt(argc, argv, "--shard", "512")), nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    const double ov = atof(opt(argc, argv, "--overlap", "0.5")), um = atof(opt(argc, argv, "--um", "2.4"));
    int S = (int)lround(P * (1.0 - ov)); if (S > P) S = P; if (S < 1) S = 1;
    if (P % 64 || P < 64) { fprintf(stderr, "surfrecto: --window must be a multiple of 64\n"); return 2; }
    if (SH % 128 || SH < 128) { fprintf(stderr, "surfrecto: --shard must be a multiple of 128\n"); return 2; }
    store *st = store_open(root); z3 *ct = st ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "surfrecto: %s %s: %s\n", root, key, z3_error()); return 1; }
    const int64_t *V = z3_meta_of(ct)->shape;
    long long bv[6]; const char *box = opt(argc, argv, "--box", nullptr);
    if (!box || sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld", bv, bv + 1, bv + 2, bv + 3, bv + 4, bv + 5) != 6) { fprintf(stderr, "surfrecto: --box z0,y0,x0,z1,y1,x1\n"); return 2; }
    ctx_t C; memset(&C, 0, sizeof C);
    C.P = P; C.SH = SH; C.nth = nth;
    for (int a = 0; a < 3; a++) {
        if (bv[a] < 0) bv[a] = 0;
        if (bv[3 + a] > V[a]) bv[3 + a] = V[a];
        if (bv[3 + a] <= bv[a]) { fprintf(stderr, "surfrecto: empty box along axis %d (volume %lld %lld %lld)\n", a, (long long)V[0], (long long)V[1], (long long)V[2]); return 2; }
        if (V[a] < P) { fprintf(stderr, "surfrecto: the volume is smaller than the window along axis %d\n", a); return 2; }
        C.box0[a] = bv[a]; C.box1[a] = bv[3 + a];
    }
    /* --part i/n: this process computes the box's i-th share of 512-row shard rows along y (n processes, e.g. one per GPU, write
       disjoint shards of the same array; each blends only its own windows) */
    int part_i = 0, part_n = 1;
    if (opt(argc, argv, "--part", nullptr) && (sscanf(opt(argc, argv, "--part", ""), "%d/%d", &part_i, &part_n) != 2 || part_n < 1 || part_i < 0 || part_i >= part_n)) {
        fprintf(stderr, "surfrecto: --part i/n with 0 <= i < n\n"); return 2;
    }
    if (part_n > 1) {
        const int64_t r0 = C.box0[1] / SH, r1 = (C.box1[1] - 1) / SH + 1, nr = r1 - r0;
        const int64_t a = r0 + nr * part_i / part_n, b = r0 + nr * (part_i + 1) / part_n;
        if (a >= b) { fprintf(stderr, "surfrecto: part %d/%d is empty (%lld shard rows)\n", part_i, part_n, (long long)nr); return 0; }
        if (a * SH > C.box0[1]) C.box0[1] = a * SH;
        if (b * SH < C.box1[1]) C.box1[1] = b * SH;
        fprintf(stderr, "surfrecto: part %d/%d: y %lld - %lld\n", part_i, part_n, (long long)C.box0[1], (long long)C.box1[1]);
    }
    for (int a = 0; a < 3; a++) C.G[a] = grid_of(C.box0[a], C.box1[a], V[a], P, S);
    const long long nwin_all = (long long)C.G[0].n * C.G[1].n * C.G[2].n;
    const double boxvox = (double)(C.box1[0] - C.box0[0]) * (C.box1[1] - C.box0[1]) * (C.box1[2] - C.box0[2]);
    fprintf(stderr, "surfrecto: CT %lldx%lldx%lld, box %lld,%lld,%lld - %lld,%lld,%lld (%.3g voxels), window %d stride %d, grid %dx%dx%d = %lld windows, tta %d\n",
            (long long)V[0], (long long)V[1], (long long)V[2], bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], boxvox, P, S, C.G[0].n, C.G[1].n, C.G[2].n, nwin_all, tta);

    /* output shards touching the box, with the linear index of the last window touching each */
    int64_t s0[3], s1[3];
    for (int a = 0; a < 3; a++) { s0[a] = C.box0[a] / SH; s1[a] = (C.box1[a] - 1) / SH + 1; }
    C.nsh = (int)((s1[0] - s0[0]) * (s1[1] - s0[1]) * (s1[2] - s0[2]));
    C.sh = calloc((size_t)C.nsh, sizeof *C.sh);
    { int i = 0;
      for (int64_t z = s0[0]; z < s1[0]; z++) for (int64_t y = s0[1]; y < s1[1]; y++) for (int64_t x = s0[2]; x < s1[2]; x++, i++) {
        shard_t *Sd = &C.sh[i]; const int64_t ix[3] = {z, y, x}; int wh[3];
        for (int a = 0; a < 3; a++) {
            Sd->idx[a] = ix[a];
            const int64_t lo = ix[a] * SH > C.box0[a] ? ix[a] * SH : C.box0[a], hi = (ix[a] + 1) * SH < C.box1[a] ? (ix[a] + 1) * SH : C.box1[a];
            Sd->o[a] = lo; Sd->n[a] = hi - lo; wh[a] = 0;
            for (int k = 0; k < C.G[a].n; k++) if (C.G[a].s[k] < hi && C.G[a].s[k] + P > lo) wh[a] = k;
        }
        Sd->last = ((long long)wh[0] * C.G[1].n + wh[1]) * C.G[2].n + wh[2];
      } }
    qsort(C.sh, (size_t)C.nsh, sizeof *C.sh, cmp_last);
    C.g1 = malloc(sizeof(float) * (size_t)P);
    { const double sg = P / 8.0; for (int i = 0; i < P; i++) C.g1[i] = (float)exp(-0.5 * (double)(i - P / 2) * (i - P / 2) / (sg * sg)); }

    /* output group / array */
    char lv[32], dir[1400], attrs[2048]; z3w_level_name(um, lv, sizeof lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"recto surface probability, uint8 = rint(255 p) (scrollprize/surface_recto_3dunet); voxel = CT level-0 voxel, 0 outside the box\","
             "\"checkpoint\":\"%s\",\"source\":\"%s/%s\",\"box_zyx\":[%lld,%lld,%lld,%lld,%lld,%lld],\"window\":%d,\"stride\":%d,\"tta\":%d,\"um\":%g}}",
             ck, root, key, bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], P, S, tta, um);
    if (z3w_write_group(out, &um, 1, "surface_recto", attrs)) { fprintf(stderr, "surfrecto: %s\n", z3w_error()); return 1; }
    snprintf(dir, sizeof dir, "%s/%s", out, lv);
    C.zw = z3w_create(dir, V, SH, 0.f, 0, attrs);
    C.sbuf = malloc((size_t)SH * SH * SH);
    if (!C.zw || !C.sbuf) { fprintf(stderr, "surfrecto: %s\n", C.zw ? "out of memory" : z3w_error()); return 1; }

    const size_t nv = (size_t)P * P * P;
    for (int s = 0; s < NSLOT; s++) { C.slots[s] = malloc(nv * 4); C.slot_free[s] = 1; if (!C.slots[s]) { fprintf(stderr, "surfrecto: out of memory\n"); return 1; } }
    pthread_mutex_init(&C.mu, nullptr); pthread_cond_init(&C.cv, nullptr);
    pthread_t bt; pthread_create(&bt, nullptr, blend_main, &C);

    reader_t R; memset(&R, 0, sizeof R);
    R.ct = ct; R.P = P; R.nth = nth; R.G = C.G; R.W = C.G[2].s[C.G[2].n - 1] + P - C.G[2].s[0]; R.nrows = C.G[0].n * C.G[1].n;
    for (int b = 0; b < 2; b++) if (!(R.buf[b] = malloc((size_t)P * P * R.W))) { fprintf(stderr, "surfrecto: out of memory (CT rows)\n"); return 1; }
    pthread_mutex_init(&R.mu, nullptr); pthread_cond_init(&R.cv, nullptr);
    pthread_t rt; pthread_create(&rt, nullptr, reader_main, &R);

    uint8_t *win = malloc(nv), *wd = xmalloc_dev(nv), *fl = tta ? xmalloc_dev(nv) : nullptr; float *tacc = tta ? xmalloc_dev(nv * 4) : nullptr;
    if (!win || !wd || (tta && (!fl || !tacc))) { fprintf(stderr, "surfrecto: out of memory\n"); return 1; }
    const char *dump = getenv("UFSM_SURFRECTO_DUMP");
    long long nwin = 0, nair = 0; double t0 = now_s(), tfwd = 0, twait = 0; int rc = 0;
    for (int r = 0; r < R.nrows && !rc; r++) {
        const int b = r & 1; const double tw = now_s();
        pthread_mutex_lock(&R.mu);
        while (!R.ready[b]) pthread_cond_wait(&R.cv, &R.mu);
        const int rerr = R.err; pthread_mutex_unlock(&R.mu);
        twait += now_s() - tw;
        if (rerr) { rc = 1; break; }
        const int iz = r / C.G[1].n, iy = r % C.G[1].n;
        for (int ix = 0; ix < C.G[2].n; ix++) {
            const long long wi = ((long long)iz * C.G[1].n + iy) * C.G[2].n + ix;
            const int64_t xo = C.G[2].s[ix] - C.G[2].s[0];
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++)
                memcpy(win + ((size_t)z * P + y) * P, R.buf[b] + ((size_t)z * P + y) * R.W + xo, P);
            double mean, sd; int vmin, vmax; sr_zstats(win, nv, &mean, &sd, &vmin, &vmax);
            job_t J = {wi, 1, nullptr, iz, iy, ix};
            if (vmin == vmax || vmax < air) { nair++; push_job(&C, J); continue; }
            const double tf = now_s();
            xh2d(wd, win, nv);
            const float *m = window_fwd(net, wd, fl, tacc, P, tta, (float)mean, (float)sd);
            if (!m) { rc = 1; break; }
            float *slot = get_slot(&C);
            xd2h(slot, m, nv * 4);
            if (dump) {   /* test hook: every window's margin, named by its CT origin */
                char nm[128]; snprintf(nm, sizeof nm, "win_%lld_%lld_%lld.f32", (long long)C.G[0].s[iz], (long long)C.G[1].s[iy], (long long)C.G[2].s[ix]);
                if (write_file(dump, nm, slot, nv * 4)) { rc = 1; break; }
            }
            tfwd += now_s() - tf; nwin++;
            J.air = 0; J.m = slot; push_job(&C, J);
            if (nwin % 25 == 0) fprintf(stderr, "surfrecto: %lld windows (+%lld air) of %lld, %.1f s, %.0f ms / window, shards written %lld (live %lld)\n",
                                        nwin, nair, nwin_all, now_s() - t0, 1e3 * tfwd / nwin, C.written, C.live);
            if (C.err) { rc = 1; break; }
        }
        pthread_mutex_lock(&R.mu); R.ready[b] = 0; pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu);
    }
    pthread_mutex_lock(&R.mu); R.stop = 1; pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu);
    pthread_join(rt, nullptr);
    pthread_mutex_lock(&C.mu); C.done = 1; pthread_cond_broadcast(&C.cv); pthread_mutex_unlock(&C.mu);
    pthread_join(bt, nullptr);
    if (C.err) rc = 1;
    if (!rc && z3w_close(C.zw)) { fprintf(stderr, "surfrecto: %s\n", z3w_error()); rc = 1; }
    const double el = now_s() - t0;
    fprintf(stderr, "surfrecto: %s: %lld windows (+%lld skipped constant / air) in %.1f s: GPU path %.0f ms / window (%.2f Mvox/s of window), CT read %.1f s (main waited %.1f s), "
                    "blend %.1f s, write %.1f s, %lld shards written (peak live %lld); box %.2f Mvox/s\n",
            rc ? "FAILED" : "done", nwin, nair, el, nwin ? 1e3 * tfwd / nwin : 0.0, nwin ? (double)nv * nwin / tfwd / 1e6 : 0.0, R.tread, twait, C.tblend, C.twrite,
            C.written, C.live_max, boxvox / el / 1e6);
    for (int b = 0; b < 2; b++) free(R.buf[b]);
    for (int s = 0; s < NSLOT; s++) free(C.slots[s]);
    for (int i = 0; i < C.nsh; i++) { free(C.sh[i].acc); free(C.sh[i].w); }
    free(C.sh); free(C.g1); free(C.sbuf); free(win); xfree_dev(wd); xfree_dev(fl); xfree_dev(tacc);
    for (int a = 0; a < 3; a++) free(C.G[a].s);
    sr_free(net); z3_close(ct);
    return rc;
}
