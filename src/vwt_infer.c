/* ufsm vwt-infer (PF): pass A of the ray model's whole-volume inference (VW's stitch design, src/vwt_tile.h): every slab of a tiling through
   the model; per slab the 4x4x2-block phase relative to its centre (OUT/phase.f16), the centre (OUT/centre.f32) and its side of every
   neighbour pair (OUT/pairs.f32), written with pwrite at the slab's / pair's offsets.

   ufsm vwt-infer CKPT ROOT KEY OUT (--tiling T.bin | --umb JSON --box z,y,x,nz,ny,nx) [--gpu 0] [--part i/n] [--threads 16] [--cache DIR]
   ufsm vwt-infer --bench N [--gpu 0]      synthetic CT band and tiling, random weights: slabs / s of the pipeline

   Parts take whole z bands round robin; each part writes NaN centres for its skipped slabs (> 98 % zero CT), so with every part run each
   slab's centre is either computed or NaN.

   Pipeline: the slabs of one z band share a CT box (the band's z range x the union of its slabs' xy ranges) read by a loader thread
   (z3, pinned host buffer, double buffered) while the GPU runs the previous band: H2D of the box, an extract + nonzero-count pass over the
   band's slabs (one sync), then per kept slab extraction on the device (trilinear, src/nn_vwi.cu), the bf16 forward, the block means and
   pair points, async D2H into a pinned record slot, and a writer thread doing the pwrites. */
#include "vwt.h"
#include "nn.h"
#include "attn.h"
#include "json.h"
#include "store.h"
#include "zarr3.h"
#include "vwt_tile.h"
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

void vwi_sample(const uint8_t *box, int bz, int by, int bx, const float *o, const float *a, const float *b, const float *d, const float *vlo, const float *vhi,
                int T, int L, uint8_t *im, uint8_t *va, void *s);
void vwi_count(const uint8_t *va, long long n, unsigned long long *cnt, void *s);
void vwi_down(const float *ph, const uint8_t *va, int T, int L, void *out, float *centre, void *s);
void vwi_points(const float *ph, const uint8_t *va, int T, int L, const float *pts, int np, float *out, void *s);
void *vwi_event_new(void);
void vwi_d2h_async(void *dst, const void *src, size_t n);
void vwi_h2d_async(void *dst, const void *src, size_t n);
void vwi_event_record(void *e);
int vwi_event_wait(void *e);

static const char *opt(int argc, char **argv, const char *k, const char *d) { for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void frame_bbox(const vwt_slab *f, int T, int L, double lo[3], double hi[3]) {   /* xyz */
    for (int e = 0; e < 3; e++) { lo[e] = 1e30; hi[e] = -1e30; }
    for (int c = 0; c < 8; c++) {
        const double ii = (c & 1) ? T - 1 : 0, jj = (c & 2) ? T - 1 : 0, kk = (c & 4) ? L - 1 : 0;
        for (int e = 0; e < 3; e++) { const double v = f->o[e] + ii * f->ax[e] + jj * f->bx[e] + kk * f->d[e]; lo[e] = fmin(lo[e], v); hi[e] = fmax(hi[e], v); }
    }
}
/* ---- band boxes: [first, last) slabs of one band, the CT box (zyx origin / size, clamped to the volume) ---- */
typedef struct { size_t first, last; int64_t bo[3], bn[3]; uint8_t *host; int ready; } band_t;
static void band_box(const vwt_tiling *S, size_t s0, size_t s1, int T, int L, const int64_t V[3], band_t *B) {
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (size_t i = s0; i < s1; i++) { double l[3], h[3]; frame_bbox(&S->s[i], T, L, l, h); for (int e = 0; e < 3; e++) { lo[e] = fmin(lo[e], l[e]); hi[e] = fmax(hi[e], h[e]); } }
    for (int e = 0; e < 3; e++) {   /* zyx from xyz */
        const int x = 2 - e; int64_t a = (int64_t)floor(lo[x]) - 1, b = (int64_t)ceil(hi[x]) + 2;
        a = a < 0 ? 0 : a; b = b > V[e] ? V[e] : b; if (b - a < 2) { a = 0; b = 2; }
        B->bo[e] = a; B->bn[e] = b - a;
    }
    B->first = s0; B->last = s1;
}

/* ---- loader thread: reads band boxes into two pinned host buffers ---- */
typedef struct {
    z3 *ct; band_t *B; size_t nb; int nth, synth; uint8_t *hbuf[2]; size_t hcap;
    pthread_mutex_t mu; pthread_cond_t cv; size_t next, done;   /* next band to read; bands [0, done) are read */
    int err, quit; size_t consumed;                              /* the GPU's band count (frees a buffer) */
} loader_t;
static void *loader_run(void *arg) {
    loader_t *Ld = arg;
    for (size_t b = 0; b < Ld->nb; b++) {
        pthread_mutex_lock(&Ld->mu);
        while (b >= Ld->consumed + 2 && !Ld->quit) pthread_cond_wait(&Ld->cv, &Ld->mu);   /* two buffers in flight */
        const int quit = Ld->quit; pthread_mutex_unlock(&Ld->mu);
        if (quit) break;
        band_t *B = &Ld->B[b]; uint8_t *h = Ld->hbuf[b & 1]; const size_t n = (size_t)B->bn[0] * B->bn[1] * B->bn[2];
        int e = 0;
        if (Ld->synth) { for (size_t i = 0; i < n; i += 4096) h[i] = (uint8_t)(i * 2654435761u >> 24); }
        else e = z3_read(Ld->ct, B->bo, B->bn, h, Ld->nth);
        pthread_mutex_lock(&Ld->mu); B->host = h; B->ready = 1; if (e) Ld->err = 1; Ld->done = b + 1; pthread_cond_broadcast(&Ld->cv); pthread_mutex_unlock(&Ld->mu);
        if (e) break;
    }
    return nullptr;
}

/* ---- writer thread: pwrites the records (phase grid + centre at the slab's offsets, the pair sides at theirs) ---- */
typedef struct { int k, np; uint8_t *data; void *ev; int *pq; } rec_t;   /* data: phase f16 [VWT_PH_N], centre f32, np x 256 f32; ready at ev */
typedef struct { int fph, fce, fpa; rec_t *q; size_t cap, head, tail; pthread_mutex_t mu; pthread_cond_t cv; int quit, err; } writer_t;
static int pw(int fd, const void *p, size_t n, off_t off) { const char *c = p; while (n) { const ssize_t w = pwrite(fd, c, n, off); if (w <= 0) return -1; c += w; n -= (size_t)w; off += w; } return 0; }
static void *writer_run(void *arg) {
    writer_t *W = arg;
    for (;;) {
        pthread_mutex_lock(&W->mu);
        while (W->head == W->tail && !W->quit) pthread_cond_wait(&W->cv, &W->mu);
        if (W->head == W->tail && W->quit) { pthread_mutex_unlock(&W->mu); break; }
        rec_t r = W->q[W->tail % W->cap]; pthread_mutex_unlock(&W->mu);
        if (vwi_event_wait(r.ev)) W->err = 1;
        if (W->fph >= 0) {
            const size_t phb = (size_t)VWT_PH_N * 2;
            int e = pw(W->fph, r.data, phb, (off_t)r.k * phb) | pw(W->fce, r.data + phb, 4, (off_t)r.k * 4);
            for (int q = 0; q < r.np; q++) e |= pw(W->fpa, r.data + phb + 4 + (size_t)q * VWT_PAIR_PTS * 4, VWT_PAIR_PTS * 4, (off_t)r.pq[q] * VWT_PAIR_PTS * 4);
            if (e) W->err = 1;
        }
        pthread_mutex_lock(&W->mu); W->tail++; pthread_cond_broadcast(&W->cv); pthread_mutex_unlock(&W->mu);
    }
    return nullptr;
}
static int open_rw(const char *dir, const char *name, size_t bytes) {   /* created at full size if missing (sparse) */
    char p[1200]; snprintf(p, sizeof p, "%s/%s", dir, name); const int fd = open(p, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return -1;
    struct stat stt; if (fstat(fd, &stt) == 0 && (size_t)stt.st_size < bytes && ftruncate(fd, (off_t)bytes)) { close(fd); return -1; }
    return fd;
}

static vwt_cfg default_cfg(int T, int L) { return (vwt_cfg){T, L, {32, 64, 96}, 192, 2, 6, 4, 256, 96, 0.04f}; }

int cmd_vwt_infer(int argc, char **argv) {
    const int bench = atoi(opt(argc, argv, "--bench", "0"));
    if (!bench && argc < 6) {
        fprintf(stderr, "usage: ufsm vwt-infer CKPT ROOT KEY OUT --umb JSON --box z,y,x,nz,ny,nx [--gpu 0] [--part i/n] [--threads 16] [--cache DIR]\n"
                        "       ufsm vwt-infer --bench N [--gpu 0]\n");
        return 2;
    }
    const int T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384")), nth = atoi(opt(argc, argv, "--threads", "16"));
    int pi = 0, pn = 1; sscanf(opt(argc, argv, "--part", "0/1"), "%d/%d", &pi, &pn); if (pn < 1 || pi < 0 || pi >= pn) { fprintf(stderr, "vwt-infer: bad --part\n"); return 2; }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) { fprintf(stderr, "vwt-infer: no GPU\n"); return 1; }
    if (!getenv("UFSM_LINEAR_MX")) at_set_mx(16);   /* the bf16 blocks */
    vwt *m; vwt_cfg c = default_cfg(T, L);
    m = bench ? vwt_create(&c, 1) : vwt_load(argv[2], 1);
    if (!m) { fprintf(stderr, "vwt-infer: no model\n"); return 1; }
    /* the tiling and the volume */
    z3 *ct = nullptr; store *st = nullptr; int64_t V[3]; vwt_tiling S = {0};
    if (bench) {
        V[0] = 240; V[1] = 2048; V[2] = 2048;
        const int64_t box[6] = {0, 0, 0, 240, 2048, 2048};
        char up[1200]; snprintf(up, sizeof up, "/tmp/vwt-infer-bench-umb-%d.json", (int)getpid());
        FILE *uf = fopen(up, "w"); if (!uf) return 1; fprintf(uf, "{\"control_points\": [{\"x\": 1024, \"y\": 1024, \"z\": 0}]}\n"); fclose(uf);
        const int rc = vwt_tiling_build(up, box, T, L, 112, 320, 112, 16, &S); unlink(up);
        if (rc) { fprintf(stderr, "vwt-infer: tiling failed\n"); return 1; }
        if (S.n > bench) S.n = bench;
    } else {
        st = store_open(argv[3]); ct = st ? z3_open(st, argv[4], opt(argc, argv, "--cache", nullptr)) : nullptr;
        if (!ct) { fprintf(stderr, "vwt-infer: %s %s: %s\n", argv[3], argv[4], z3_error()); return 1; }
        memcpy(V, z3_meta_of(ct)->shape, sizeof V);
        if (*opt(argc, argv, "--tiling", "")) { if (vwt_tiling_load(opt(argc, argv, "--tiling", ""), &S)) { fprintf(stderr, "vwt-infer: cannot load the tiling\n"); return 1; } }
        else {
            long long bx[6]; int64_t box[6];
            if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &bx[0], &bx[1], &bx[2], &bx[3], &bx[4], &bx[5]) != 6) { fprintf(stderr, "vwt-infer: --tiling or --umb + --box\n"); return 2; }
            for (int e = 0; e < 6; e++) box[e] = bx[e];
            if (vwt_tiling_build(opt(argc, argv, "--umb", ""), box, T, L, 112, 320, 112, 16, &S)) { fprintf(stderr, "vwt-infer: tiling failed\n"); return 1; }
        }
        if (S.T != T || S.L != L) { fprintf(stderr, "vwt-infer: tiling T %d L %d vs model %d %d\n", S.T, S.L, T, L); return 1; }
    }
    /* the pairs of every slab (CSR; vwt_slab_pairs is a scan per slab) */
    int *poff = calloc((size_t)S.n + 1, sizeof *poff), *pidx = malloc(((size_t)S.np * 2 + 1) * sizeof *pidx);
    for (int q = 0; q < S.np; q++) { if (S.p[q].i < S.n) poff[S.p[q].i + 1]++; if (S.p[q].j < S.n) poff[S.p[q].j + 1]++; }
    for (int k = 0; k < S.n; k++) poff[k + 1] += poff[k];
    { int *fill = calloc((size_t)S.n + 1, sizeof *fill);
      for (int q = 0; q < S.np; q++) for (int sd = 0; sd < 2; sd++) { const int k = sd ? S.p[q].j : S.p[q].i; if (k < S.n) pidx[poff[k] + fill[k]++] = 2 * q + sd; }
      free(fill); }
    int maxp = 0; for (int k = 0; k < S.n; k++) if (poff[k + 1] - poff[k] > maxp) maxp = poff[k + 1] - poff[k];
    /* this part's slabs: whole bands, round robin (a band's CT is read once) */
    size_t nbands = 0; band_t *B = calloc((size_t)S.n + 1, sizeof *B);
    { size_t s0 = 0; int bi = 0;
      for (size_t i = 1; i <= (size_t)S.n; i++) if (i == (size_t)S.n || S.s[i].b != S.s[s0].b) { if (bi++ % pn == pi) band_box(&S, s0, i, T, L, V, &B[nbands++]); s0 = i; } }
    size_t nsl = 0, hmax = 0; for (size_t b = 0; b < nbands; b++) { nsl += B[b].last - B[b].first; const size_t n = (size_t)B[b].bn[0] * B[b].bn[1] * B[b].bn[2]; if (n > hmax) hmax = n; }
    fprintf(stderr, "vwt-infer: %d slabs, %d pairs (<= %d per slab); part %d/%d: %zu bands, %zu slabs, largest band box %.0f MB\n", S.n, S.np, maxp, pi, pn, nbands, nsl, hmax / 1e6);
    /* buffers */
    const size_t nv = (size_t)T * T * L, phb = (size_t)VWT_PH_N * 2, rbytes = phb + 4 + (size_t)maxp * VWT_PAIR_PTS * 4, rstride = (rbytes + 255) & ~(size_t)255;
    uint8_t *dbox = nn_malloc(hmax ? hmax : 1), *imd = nn_malloc(nv), *vad = nn_malloc(nv); float *phd = nn_malloc(nv * 4);
    float *ptsd = nn_malloc((size_t)(maxp ? maxp : 1) * VWT_PAIR_PTS * 3 * 4), *ptsh = nn_host_alloc((size_t)(maxp ? maxp : 1) * VWT_PAIR_PTS * 3 * 4);
    unsigned long long *cnts = nullptr, *hcnt = nullptr; size_t ccap = 0;   /* per-slab nonzero counts of a band (device, host) */
    loader_t Ld = {ct, B, nbands, nth, bench, {nn_host_alloc(hmax ? hmax : 1), nn_host_alloc(hmax ? hmax : 1)}, hmax, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, 0, 0};
    if (bench) for (int k = 0; k < 2; k++) memset(Ld.hbuf[k], 90, hmax);
    writer_t W = {-1, -1, -1, nullptr, 8, 0, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0};
    W.q = calloc(W.cap, sizeof *W.q); uint8_t *wbuf = nn_host_alloc(W.cap * rstride), *outd = nn_malloc(W.cap * rstride);   /* record slots: device and pinned */
    for (size_t q = 0; q < W.cap; q++) W.q[q].pq = malloc((size_t)(maxp ? maxp : 1) * sizeof(int));
    if (!bench) {
        const char *out = argv[5]; mkdir(out, 0755);
        W.fph = open_rw(out, "phase.f16", (size_t)S.n * phb); W.fce = open_rw(out, "centre.f32", (size_t)S.n * 4); W.fpa = open_rw(out, "pairs.f32", (size_t)S.np * 2 * VWT_PAIR_PTS * 4);
        if (W.fph < 0 || W.fce < 0 || W.fpa < 0) { fprintf(stderr, "vwt-infer: cannot open the outputs in %s\n", out); return 1; }
        char p[1200]; snprintf(p, sizeof p, "%s/tiling.bin", out); if (access(p, F_OK) && vwt_tiling_save(&S, p)) { fprintf(stderr, "vwt-infer: cannot write %s\n", p); return 1; }
    }
    pthread_t lt, wt; pthread_create(&lt, nullptr, loader_run, &Ld); pthread_create(&wt, nullptr, writer_run, &W);
    double t0 = now_s(), tload = 0, tgpu = 0; size_t nrun = 0, nskip = 0; void *ptsev = nullptr;
    for (size_t b = 0; b < nbands; b++) {
        const double tw = now_s();
        pthread_mutex_lock(&Ld.mu); while (Ld.done <= b && !Ld.err) pthread_cond_wait(&Ld.cv, &Ld.mu); const int err = Ld.err; pthread_mutex_unlock(&Ld.mu);
        if (err) { fprintf(stderr, "vwt-infer: CT read: %s\n", z3_error()); return 1; }
        tload += now_s() - tw;
        band_t *bb = &B[b]; const size_t bnb = (size_t)bb->bn[0] * bb->bn[1] * bb->bn[2];
        nn_h2d(dbox, bb->host, bnb);
        pthread_mutex_lock(&Ld.mu); Ld.consumed = b + 1; pthread_cond_broadcast(&Ld.cv); pthread_mutex_unlock(&Ld.mu);
        const double tg = now_s();
        const size_t ns = bb->last - bb->first;
        if (ns > ccap) { nn_free(cnts); cnts = nn_malloc(ns * 8); free(hcnt); hcnt = malloc(ns * 8); ccap = ns; }
        for (int pass = 0; pass < 2; pass++)   /* 0: extract + count every slab (one sync), 1: the kept slabs through the model */
        for (size_t i = bb->first; i < bb->last; i++) {
            const vwt_slab *s = &S.s[i]; float o[3], a[3], bv[3], d[3], vlo[3], vhi[3];
            if (pass && !bench && hcnt[i - bb->first] < nv / 50) {   /* > 98 % zero CT: centre NaN */
                const float nanv = NAN; if (pw(W.fce, &nanv, 4, (off_t)i * 4)) { fprintf(stderr, "vwt-infer: write failed\n"); return 1; }
                nskip++; continue;
            }
            for (int e = 0; e < 3; e++) {   /* xyz relative to the box origin (zyx bo) */
                const int z = 2 - e; o[e] = (float)(s->o[e] - bb->bo[z]); a[e] = (float)s->ax[e]; bv[e] = (float)s->bx[e]; d[e] = (float)s->d[e];
                vlo[e] = (float)(-bb->bo[z]); vhi[e] = (float)(V[z] - 1 - bb->bo[z]);
            }
            vwi_sample(dbox, (int)bb->bn[0], (int)bb->bn[1], (int)bb->bn[2], o, a, bv, d, vlo, vhi, T, L, imd, vad, 0);
            if (!pass) {
                vwi_count(imd, (long long)nv, cnts + (i - bb->first), 0);
                if (i + 1 == bb->last) nn_d2h(hcnt, cnts, ns * 8);
                continue;
            }
            if (vwt_forward(m, imd, vad, phd, nullptr, 0)) return 1;
            /* the record: wait for a free writer slot (its device and host buffers), then the outputs into it, asynchronously */
            pthread_mutex_lock(&W.mu); while (W.head - W.tail >= W.cap) pthread_cond_wait(&W.cv, &W.mu); pthread_mutex_unlock(&W.mu);
            const size_t slot = W.head % W.cap; rec_t *r = &W.q[slot]; uint8_t *od = outd + slot * rstride;
            vwi_down(phd, vad, T, L, od, (float *)(od + phb), 0);
            /* the pair points of this slab's sides, slab-local (the host buffer is reused: wait for the previous upload) */
            const int np_ = poff[i + 1] - poff[i];
            if (np_) {
                if (ptsev) vwi_event_wait(ptsev);
                for (int q = 0; q < np_; q++) {
                    const int pq = pidx[poff[i] + q]; const vwt_pair *P = &S.p[pq >> 1]; r->pq[q] = pq;
                    for (int e = 0; e < VWT_PAIR_PTS; e++) {
                        float *u = ptsh + ((size_t)q * VWT_PAIR_PTS + e) * 3;
                        if (e < P->npts) { const double w3[3] = {P->pts[e][0], P->pts[e][1], P->pts[e][2]}; double uu[3]; vwt_to_slab(s, w3, uu); u[0] = (float)uu[0]; u[1] = (float)uu[1]; u[2] = (float)uu[2]; }
                        else u[0] = u[1] = u[2] = -1e9f;   /* (NaN out) */
                    }
                }
                vwi_h2d_async(ptsd, ptsh, (size_t)np_ * VWT_PAIR_PTS * 12);
                if (!ptsev) ptsev = vwi_event_new();
                vwi_event_record(ptsev);
                vwi_points(phd, vad, T, L, ptsd, np_ * VWT_PAIR_PTS, (float *)(od + phb + 4), 0);
            }
            r->k = (int)i; r->np = np_; r->data = wbuf + slot * rstride;
            if (!r->ev) r->ev = vwi_event_new();
            vwi_d2h_async(r->data, od, phb + 4 + (size_t)np_ * VWT_PAIR_PTS * 4); vwi_event_record(r->ev);
            pthread_mutex_lock(&W.mu); W.head++; pthread_cond_broadcast(&W.cv); pthread_mutex_unlock(&W.mu);
            nrun++;
            if (nrun % 200 == 0) { const double el = now_s() - t0; fprintf(stderr, "vwt-infer: %zu / %zu slabs (%zu skipped), %.0f ms / slab, CT wait %.1f s\n", nrun, nsl, nskip, el / nrun * 1e3, tload); }
        }
        nn_sync(); tgpu += now_s() - tg;
    }
    pthread_mutex_lock(&W.mu); W.quit = 1; pthread_cond_broadcast(&W.cv); pthread_mutex_unlock(&W.mu); pthread_join(wt, nullptr);
    pthread_mutex_lock(&Ld.mu); Ld.quit = 1; pthread_cond_broadcast(&Ld.cv); pthread_mutex_unlock(&Ld.mu); pthread_join(lt, nullptr);
    if (W.fph >= 0) { close(W.fph); close(W.fce); close(W.fpa); }
    const double el = now_s() - t0;
    printf("vwt-infer: %zu slabs (%zu skipped) in %.1f s = %.1f ms / slab (GPU %.1f s, CT wait %.1f s); %.2f slabs / s\n", nrun, nskip, el, nrun ? el / nrun * 1e3 : 0, tgpu, tload, nrun / el);
    nn_host_free(Ld.hbuf[0]); nn_host_free(Ld.hbuf[1]); nn_host_free(wbuf);
    if (ct) z3_close(ct);
    return W.err ? 1 : 0;
}
