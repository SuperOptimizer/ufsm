/* band_w4dev.h: the device side of band_wind4_dev. Raw CUDA with its own streams and buffers (the sampler's worker threads call
   it beside the trainer's kernels; nothing of nn_core's per-trainer state is touched). Integer kernels only: the bytes are
   band_wind4_tiled's.
   A window is processed in strips of one tile in z and y (the tiles are independent; a strip holds every tile of its x row):
   <= T * T * W voxels on the device at a time (4 B per voxel + the events).
     w4_runs_k    one thread per row of a tile: the runs along x (w4_row_joins' rule), per voxel its run's ordinal in the row and
                  its turns since the run's start
     w4_events_k  one thread per row: per run its length and its events, in x order (count pass, then the emit pass at the
                  offsets the host's prefix sums give: tiles in raster order, rows in scan order = the CPU's run numbering)
     w4_out_k     per voxel the winding code and the component from its run's root offset / component byte (w4_pass1) */
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "band_w4dev.h"

namespace {
enum { NCTX = 2, MAXDEV = 8, UNK = 0xFFFF };

__host__ __device__ __forceinline__ int w4_joined(int va, int vb, int ms, int *t) {   /* w4_row_joins: joined, *t the turn the step adds (-1, 0, 1) */
    const int d = vb - va, st = d >= 126 ? d - 252 : d < -126 ? d + 252 : d;
    *t = (st > d) - (st < d);
    return va < 252 && vb < 252 && st <= ms && st >= -ms;
}
/* strip [nz][ny][W], tiles of T along x; row r = (z * ny + y) * ntx + tx */
__host__ __device__ __forceinline__ void w4_runs_row(size_t r, const uint8_t *ph, int nz, int ny, int W, int T, int ntx, int ms, uint16_t *R, uint32_t *nrun) {   /* (one row: w4_runs_k, the CPU fallback) */
    const int tx = (int)(r % ntx), x0 = tx * T, x1 = min(W, x0 + T); const size_t row = (r / ntx) * W;
    int ord = -1, j = 0, prev = 255;
    for (int x = x0; x < x1; x++) {
        const int v = ph[row + x];
        if (v >= 252) { R[row + x] = UNK; prev = 255; continue; }
        int t;
        if (w4_joined(prev, v, ms, &t)) j = (j + t) & 3; else { ord++; j = 0; }
        R[row + x] = (uint16_t)(ord << 2 | j); prev = v;
    }
    nrun[r] = (uint32_t)(ord + 1);
}
__global__ void w4_runs_k(const uint8_t *ph, int nz, int ny, int W, int T, int ntx, int ms, uint16_t *R, uint32_t *nrun) {
    const size_t r = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (r < (size_t)nz * ny * ntx) w4_runs_row(r, ph, nz, ny, W, T, ntx, ms, R, nrun);
}
__host__ __device__ __forceinline__ void w4_events_row(size_t r, const uint8_t *ph, const uint16_t *R, int nz, int ny, int W, int T, int ntx, int ms, int emit, const uint32_t *runoff, const uint32_t *evoff,
                                                      uint32_t *nev, w4d_run *runs, w4d_ev *ev) {   /* (one row: w4_events_k, the CPU fallback) */
    const int tx = (int)(r % ntx), x0 = tx * T, x1 = min(W, x0 + T), y = (int)(r / ntx % ny), z = (int)(r / ntx / ny); const size_t row = (r / ntx) * W, pl = (size_t)ny * W;
    const int has[2] = {y > 0, z > 0}; const size_t nb[2] = {row - W, row - pl}, nr[2] = {r - ntx, r - (size_t)ny * ntx};   /* the y / z neighbour rows */
    const uint32_t ro = emit ? runoff[r] : 0, eo = emit ? evoff[r] : 0;
    int cur = -1, len = 0, nevr = 0, c[2] = {-1, -1}; uint32_t ne = 0, oi[2] = {0, 0}; uint16_t h[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    auto flush = [&](int s) { if (c[s] >= 0 && emit) { w4d_ev *e = ev + eo + oi[s]; e->h[0] = h[s][0]; e->h[1] = h[s][1]; e->h[2] = h[s][2]; e->h[3] = h[s][3]; } c[s] = -1; };
    auto close = [&]() { flush(0); flush(1); if (cur >= 0 && emit) { runs[ro + cur].len = (uint16_t)len; runs[ro + cur].nev = (uint16_t)nevr; } };
    for (int x = x0; x < x1; x++) {
        const uint16_t q = R[row + x];
        if (q == UNK) continue;
        const int ord = q >> 2, j = q & 3, v = ph[row + x];
        if (ord != cur) { close(); cur = ord; len = 0; nevr = 0; }
        len++;
        for (int s = 0; s < 2; s++) {
            if (!has[s]) continue;
            const uint16_t qn = R[nb[s] + x]; int t;
            if (qn == UNK || !w4_joined(ph[nb[s] + x], v, ms, &t)) continue;
            const int on = qn >> 2, rel = (t + (qn & 3) - j) & 3;
            if (on == c[s]) { h[s][rel]++; continue; }
            flush(s);
            c[s] = on; oi[s] = ne++; nevr++; h[s][0] = h[s][1] = h[s][2] = h[s][3] = 0;
            if (emit) { w4d_ev *e = ev + eo + oi[s]; e->rn = runoff[nr[s]] + (uint32_t)on; e->xrel = (uint16_t)len; e->rel = (uint8_t)(rel | s << 2); e->pad = 0; }
        }
    }
    close();
    if (!emit) nev[r] = ne;
}
__global__ void w4_events_k(const uint8_t *ph, const uint16_t *R, int nz, int ny, int W, int T, int ntx, int ms, int emit, const uint32_t *runoff, const uint32_t *evoff,
                            uint32_t *nev, w4d_run *runs, w4d_ev *ev) {
    const size_t r = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (r < (size_t)nz * ny * ntx) w4_events_row(r, ph, R, nz, ny, W, T, ntx, ms, emit, runoff, evoff, nev, runs, ev);
}
__host__ __device__ __forceinline__ void w4_out_vox(size_t i, uint8_t *ph, const uint16_t *R, int nz, int ny, int W, int T, int ntx, const uint32_t *runoff, const uint8_t *rro, const uint8_t *rcb, uint8_t *comp) {
    const uint16_t q = R[i]; uint8_t w = 255, c = 255;
    if (q != UNK) {
        const size_t r = (i / W) * ntx + (i % W) / T; const uint32_t run = runoff[r] + (q >> 2);
        c = rcb[run];
        if (c != 255) w = (uint8_t)(((ph[i] + 252 * ((q + rro[run]) & 3) + 2) / 4) % 252);   /* (w4_tab) */
    }
    ph[i] = w; comp[i] = c;
}
__global__ void w4_out_k(uint8_t *ph, const uint16_t *R, int nz, int ny, int W, int T, int ntx, const uint32_t *runoff, const uint8_t *rro, const uint8_t *rcb, uint8_t *comp) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)nz * ny * W) w4_out_vox(i, ph, R, nz, ny, W, T, ntx, runoff, rro, rcb, comp);
}

/* --- the host side: band.c's union-find (w4_find / w4_join / the component choice of band_wind4) replayed over the run events. A
   copy, not a call: band.c's are static and tied to its voxel loop. One tile: its runs (scan order) and their events (run order,
   each run's in x order, y before z; ev[].rn - rn0 = the neighbour's index in the tile). Writes per run ro (turns to its root) and
   cid (the tile's component; both 255: none kept); returns the components kept (<= maxcomp <= 15), -1 out of memory. st: added to. --- */
typedef struct { int32_t parent; uint8_t off; } wr_run;
typedef struct { wr_run *r; int64_t *size, *edges, *conf; int16_t *map; size_t cap; } wr_uf;
thread_local wr_uf t_uf;   /* (kept by the thread between tiles: no realloc growth per tile) */

int32_t wr_find(wr_uf *u, int32_t a, int *off) {   /* = w4_find */
    int o = 0; int32_t x = a;
    while (u->r[x].parent != x) { o += u->r[x].off; x = u->r[x].parent; }
    const int32_t root = x; int acc = o;
    x = a;
    while (u->r[x].parent != root && u->r[x].parent != x) {
        const int32_t p = u->r[x].parent; const int ox = u->r[x].off;
        u->r[x].parent = root; u->r[x].off = (uint8_t)(acc & 3); acc -= ox; x = p;
    }
    *off = o & 3; return root;
}
int wr_join(wr_uf *u, int32_t cur, int32_t rn, int rel) {   /* = w4_join */
    int oa, ob; const int32_t ra = wr_find(u, cur, &oa), rb = wr_find(u, rn, &ob);
    if (ra == rb) { u->edges[ra]++; if (((oa - ob - rel) & 3) != 0) u->conf[ra]++; return (oa - ob) & 3; }
    const int d = (rel - oa + ob) & 3;
    if (u->size[ra] < u->size[rb]) { u->r[ra].parent = rb; u->r[ra].off = (uint8_t)d; u->size[rb] += u->size[ra]; u->edges[rb] += u->edges[ra] + 1; u->conf[rb] += u->conf[ra]; }
    else { u->r[rb].parent = ra; u->r[rb].off = (uint8_t)((-d) & 3); u->size[ra] += u->size[rb]; u->edges[ra] += u->edges[rb] + 1; u->conf[ra] += u->conf[rb]; }
    return rel;
}
int replay(const w4d_run *runs, size_t nruns, const w4d_ev *ev, size_t nev, uint32_t rn0, size_t minsize, int maxcomp, uint8_t *ro, uint8_t *cid, band_wind4_stats *st) {
    wr_uf *u = &t_uf;
    if (maxcomp > 15) maxcomp = 15;
    if (nruns > u->cap) {
        const size_t cap = nruns + nruns / 4 + 4096;
        free(u->r); free(u->size); free(u->edges); free(u->conf); free(u->map);
        u->r = (wr_run *)malloc(cap * sizeof *u->r); u->size = (int64_t *)malloc(cap * 8); u->edges = (int64_t *)malloc(cap * 8); u->conf = (int64_t *)malloc(cap * 8); u->map = (int16_t *)malloc(cap * sizeof *u->map);
        if (!u->r || !u->size || !u->edges || !u->conf || !u->map) { free(u->r); free(u->size); free(u->edges); free(u->conf); free(u->map); *u = wr_uf{}; return -1; }
        u->cap = cap;
    }
    size_t e = 0;
    for (size_t r = 0; r < nruns; r++) {
        u->r[r].parent = (int32_t)r; u->r[r].off = 0; u->size[r] = 0; u->edges[r] = 0; u->conf[r] = 0;
        int added = 0, o;
        for (int k = 0; k < runs[r].nev; k++, e++) {
            if (e >= nev) return -1;
            const w4d_ev *v = ev + e;
            /* the run's voxels up to this one are in its component's size before a union that can merge (the size comparison) */
            if (v->xrel != added) { u->size[wr_find(u, (int32_t)r, &o)] += v->xrel - added; added = v->xrel; }
            const int d = wr_join(u, (int32_t)r, (int32_t)(v->rn - rn0), v->rel & 3);
            const int cnt = v->h[0] + v->h[1] + v->h[2] + v->h[3];   /* the further voxels joined to the same run: edges inside one component, conflicts where their turn differs */
            if (cnt) { const int32_t root = wr_find(u, (int32_t)r, &o); u->edges[root] += cnt; u->conf[root] += cnt - v->h[d]; }
        }
        if (runs[r].len != added) u->size[wr_find(u, (int32_t)r, &o)] += runs[r].len - added;
    }
    /* the components kept: band_wind4's choice */
    int32_t best[15]; int nb = 0; size_t known = 0, kept = 0, dropped = 0;
    for (size_t i = 0; i < nruns; i++) {
        u->map[i] = -1;
        if (u->r[i].parent != (int32_t)i) continue;
        known += (size_t)u->size[i];
        if ((size_t)u->size[i] < minsize) continue;
        if (u->conf[i] > 8 + u->edges[i] / 500) { dropped += (size_t)u->size[i]; continue; }
        int p = nb < maxcomp ? nb++ : maxcomp;
        if (p == maxcomp && u->size[i] <= u->size[best[maxcomp - 1]]) continue;
        if (p == maxcomp) p = maxcomp - 1;
        while (p > 0 && u->size[best[p - 1]] < u->size[i]) { best[p] = best[p - 1]; p--; }
        best[p] = (int32_t)i;
    }
    for (int c = 0; c < nb; c++) { u->map[best[c]] = (int16_t)c; kept += (size_t)u->size[best[c]]; }
    for (size_t r = 0; r < nruns; r++) {
        int off; const int32_t root = wr_find(u, (int32_t)r, &off); const int16_t c = u->map[root];
        ro[r] = c < 0 ? 255 : (uint8_t)off; cid[r] = c < 0 ? 255 : (uint8_t)c;
    }
    if (st) { st->known += known; st->kept += kept; st->dropped_conflict += dropped; st->runs += nruns; }
    return nb;
}

struct w4ctx {
    int dev, ready; pthread_mutex_t mu; cudaStream_t st;
    uint8_t *d_ph, *d_c; uint16_t *d_R; size_t vox;                 /* strip planes */
    uint32_t *d_cnt, *d_off, *h_cnt, *h_off; size_t rows;            /* per row: runs, events (counts; offsets) */
    w4d_run *d_runs, *h_runs; uint8_t *d_lut, *h_lut; size_t nruns;  /* lut: [ro | cb] */
    w4d_ev *d_ev, *h_ev; size_t nev;
};
w4ctx g_ctx[MAXDEV * NCTX]; int g_nctx; volatile int g_next;
/* g_bad: after a device failure (no memory beside the trainer: 2026-11-12 unissl, a full 16 GB card) the next W4_COOL calls take the
   CPU, then the device is tried again -- before, one failure moved the rest of the run to the CPU (or, in nn_w4runs_label, aborted) */
constexpr int W4_COOL = 256; volatile int g_bad;
static bool w4_dev_ok(void) { if (g_bad <= 0) return true; __sync_fetch_and_sub(&g_bad, 1); return false; }
static void w4_dev_failed(const char *what, int dev) {
    g_bad = W4_COOL; static int n; const int k = __sync_add_and_fetch(&n, 1);
    if (k <= 3 || !(k & (k - 1))) fprintf(stderr, "%s: CUDA error or no device memory on GPU %d (%s): the CPU for the next %d calls, then the device again (%d failures)\n", what, dev, cudaGetErrorString(cudaGetLastError()), W4_COOL, k);
    else cudaGetLastError();
}
/* tests: the next n device runs (ctx_run, nn_w4runs_apply) fail as if out of memory (the fallbacks' paths) */
static volatile int g_w4_inject;
static bool w4_inj(void) { return g_w4_inject > 0 && __sync_sub_and_fetch(&g_w4_inject, 1) >= 0; }

#define CK(x) do { if ((x) != cudaSuccess) return -2; } while (0)
/* cudaStreamSynchronize spins a core while the stream waits behind the trainer's kernels: poll with short sleeps instead (the
   worker's CPU time is what this path is for) */
cudaError_t w4_wait(cudaStream_t st) {
    for (;;) {
        const cudaError_t e = cudaStreamQuery(st);
        if (e != cudaErrorNotReady) return e;
        const struct timespec ts = {0, 100000}; nanosleep(&ts, nullptr);
    }
}
template <typename T> int dgrow(T **p, size_t *have, size_t need, size_t el) {
    (void)have; if (*p) cudaFree(*p); *p = nullptr;
    return cudaMalloc((void **)p, need * el) == cudaSuccess ? 0 : -2;
}
template <typename T> int hgrow(T **p, size_t need, size_t el) {
    if (*p) cudaFreeHost(*p); *p = nullptr;
    return cudaHostAlloc((void **)p, need * el, cudaHostAllocDefault) == cudaSuccess ? 0 : -2;
}
/* w4runs: one strip's run offsets (+ base: the runs of the strips before) and LUT appended */
int runs_append(w4runs *o, const uint32_t *off, size_t rows, uint32_t base, const uint8_t *ro, const uint8_t *cb, size_t nr) {
    if (o->noff + rows > o->cap_off) {
        const size_t cap = (o->noff + rows) * 5 / 4 + 1024; uint32_t *p = (uint32_t *)realloc(o->off, cap * sizeof *p); if (!p) return -1; o->off = p; o->cap_off = cap;
    }
    if (o->nr + nr > o->cap_r) {
        const size_t cap = (o->nr + nr) * 5 / 4 + 4096; uint8_t *a = (uint8_t *)realloc(o->ro, cap); if (!a) return -1; o->ro = a;
        uint8_t *b = (uint8_t *)realloc(o->cb, cap); if (!b) return -1; o->cb = b; o->cap_r = cap;
    }
    for (size_t r = 0; r < rows; r++) o->off[o->noff + r] = off[r] + base;
    memcpy(o->ro + o->nr, ro, nr); memcpy(o->cb + o->nr, cb, nr);
    o->noff += rows; o->nr += nr;
    return 0;
}
/* the tile geometry band_wind4_dev uses (one: the whole window as one tile) */
void w4_geom(int D, int H, int W, int tile, int *one, int *T, int *ntx) {
    *one = tile <= 0 || (tile >= D && tile >= H && tile >= W); *T = *one ? (D > H ? (D > W ? D : W) : (H > W ? H : W)) : tile; *ntx = (W + *T - 1) / *T;
}
/* the runs path on the CPU (no device): the row bodies of w4_runs_k / w4_events_k in loops, the same offsets, replay and numbering */
int cpu_runs(const uint8_t *phase, int D, int H, int W, int tile, int ms, size_t minsize, int maxcomp, w4runs *o, band_wind4_stats *st) {
    int one, T, ntx; w4_geom(D, H, W, tile, &one, &T, &ntx);
    if (T >= 16384 || ntx > 1023) return -1;
    const int mc = maxcomp > 15 ? 15 : maxcomp;
    int next = 0; band_wind4_stats acc = {0}; o->noff = o->nr = 0;
    const size_t sv = (size_t)(T < D ? T : D) * (T < H ? T : H) * W, sr = (size_t)(T < D ? T : D) * (T < H ? T : H) * ntx;
    uint8_t *ph = (uint8_t *)malloc(sv); uint16_t *R = (uint16_t *)malloc(sv * 2); uint32_t *cnt = (uint32_t *)malloc(2 * sr * 4), *off = (uint32_t *)malloc(2 * sr * 4);
    w4d_run *runs = nullptr; w4d_ev *ev = nullptr; uint8_t *lut = nullptr; size_t capr = 0, cape = 0; int rc = 0; uint32_t gbase = 0;
    if (!ph || !R || !cnt || !off) rc = -1;
    for (int z0 = 0; z0 < D && !rc; z0 += T) for (int y0 = 0; y0 < H && !rc; y0 += T) {
        const int nz = D - z0 < T ? D - z0 : T, ny = H - y0 < T ? H - y0 : T;
        const size_t rows = (size_t)nz * ny * ntx;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) memcpy(ph + ((size_t)z * ny + y) * W, phase + ((size_t)(z0 + z) * H + y0 + y) * W, (size_t)W);
        for (size_t r = 0; r < rows; r++) w4_runs_row(r, ph, nz, ny, W, T, ntx, ms, R, cnt);
        for (size_t r = 0; r < rows; r++) w4_events_row(r, ph, R, nz, ny, W, T, ntx, ms, 0, nullptr, nullptr, cnt + rows, nullptr, nullptr);
        size_t nr = 0, ne = 0, tr0[1025], te0[1025];
        for (int tx = 0; tx < ntx; tx++) {
            tr0[tx] = nr; te0[tx] = ne;
            for (size_t zy = 0; zy < (size_t)nz * ny; zy++) { const size_t r = zy * ntx + tx; off[r] = (uint32_t)nr; off[rows + r] = (uint32_t)ne; nr += cnt[r]; ne += cnt[rows + r]; }
        }
        tr0[ntx] = nr; te0[ntx] = ne;
        if (nr > 0xfffffff0u || ne > 0xfffffff0u || (uint64_t)gbase + nr > 0xfffffff0u) { rc = -1; break; }
        if (nr > capr) { capr = nr + nr / 4 + 4096; free(runs); free(lut); runs = (w4d_run *)malloc(capr * sizeof *runs); lut = (uint8_t *)malloc(2 * capr); if (!runs || !lut) { rc = -1; break; } }
        if (ne > cape) { cape = ne + ne / 4 + 4096; free(ev); ev = (w4d_ev *)malloc(cape * sizeof *ev); if (!ev) { rc = -1; break; } }
        for (size_t r = 0; r < rows; r++) w4_events_row(r, ph, R, nz, ny, W, T, ntx, ms, 1, off, off + rows, nullptr, runs, ev);
        uint8_t *ro = lut, *cb = lut + nr;
        for (int tx = 0; tx < ntx && !rc; tx++) {
            const size_t a = tr0[tx], n = tr0[tx + 1] - a;
            const int nc = replay(runs + a, n, ev + te0[tx], te0[tx + 1] - te0[tx], (uint32_t)a, minsize, mc, ro + a, cb + a, &acc);
            if (nc < 0) { rc = -1; break; }
            const int base = next; next += nc;
            if (!one) for (size_t k = a; k < a + n; k++) { if (cb[k] == 255 || base + cb[k] > 254) { cb[k] = 255; ro[k] = 255; } else cb[k] = (uint8_t)(base + cb[k]); }
        }
        if (!rc && runs_append(o, off, rows, gbase, ro, cb, nr)) rc = -1;
        gbase += (uint32_t)nr;
    }
    free(ph); free(R); free(cnt); free(off); free(runs); free(ev); free(lut);
    if (rc) return -1;
    if (!one && next > 255) next = 255;
    acc.ncomp = next; o->ncomp = next;
    if (st) *st = acc;
    return next;
}
int ctx_run(w4ctx *c, const uint8_t *phase, int D, int H, int W, int tile, int ms, size_t minsize, int maxcomp, uint8_t *w4, uint8_t *comp, band_wind4_stats *st, w4runs *ro_out = nullptr) {
    uint32_t gbase = 0; if (ro_out) ro_out->noff = ro_out->nr = 0;
    if (w4_inj()) return -2;
    CK(cudaSetDevice(c->dev));
    if (!c->ready) { CK(cudaStreamCreate(&c->st)); c->ready = 1; }
    const int one = tile <= 0 || (tile >= D && tile >= H && tile >= W), T = one ? (D > H ? (D > W ? D : W) : (H > W ? H : W)) : tile, ntx = (W + T - 1) / T;
    const int mc = maxcomp > 15 ? 15 : maxcomp;
    int next = 0; band_wind4_stats acc = {0};
    for (int z0 = 0; z0 < D; z0 += T) for (int y0 = 0; y0 < H; y0 += T) {
        const int nz = D - z0 < T ? D - z0 : T, ny = H - y0 < T ? H - y0 : T;
        const size_t vox = (size_t)nz * ny * W, rows = (size_t)nz * ny * ntx, off0 = ((size_t)z0 * H + y0) * W;
        if (vox > c->vox) {
            c->vox = 0; if (dgrow(&c->d_ph, nullptr, vox, 1) || dgrow(&c->d_c, nullptr, vox, 1) || dgrow(&c->d_R, nullptr, vox, 2)) return -2; c->vox = vox;
        }
        if (rows > c->rows) {
            c->rows = 0; if (dgrow(&c->d_cnt, nullptr, 2 * rows, 4) || dgrow(&c->d_off, nullptr, 2 * rows, 4) || hgrow(&c->h_cnt, 2 * rows, 4) || hgrow(&c->h_off, 2 * rows, 4)) return -2; c->rows = rows;
        }
        CK(cudaMemcpy2DAsync(c->d_ph, (size_t)ny * W, phase + off0, (size_t)H * W, (size_t)ny * W, nz, cudaMemcpyHostToDevice, c->st));
        const unsigned rb = (unsigned)((rows + 127) / 128);
        w4_runs_k<<<rb, 128, 0, c->st>>>(c->d_ph, nz, ny, W, T, ntx, ms, c->d_R, c->d_cnt);
        w4_events_k<<<rb, 128, 0, c->st>>>(c->d_ph, c->d_R, nz, ny, W, T, ntx, ms, 0, nullptr, nullptr, c->d_cnt + rows, nullptr, nullptr);
        CK(cudaMemcpyAsync(c->h_cnt, c->d_cnt, 2 * rows * 4, cudaMemcpyDeviceToHost, c->st));
        CK(w4_wait(c->st));
        /* offsets: the tiles of the strip in x order, each tile's rows in scan order (z, y) */
        size_t nr = 0, ne = 0; size_t tr0[1024], te0[1024];
        if (ntx > 1023) return -2;
        for (int tx = 0; tx < ntx; tx++) {
            tr0[tx] = nr; te0[tx] = ne;
            for (size_t zy = 0; zy < (size_t)nz * ny; zy++) { const size_t r = zy * ntx + tx; c->h_off[r] = (uint32_t)nr; c->h_off[rows + r] = (uint32_t)ne; nr += c->h_cnt[r]; ne += c->h_cnt[rows + r]; }
        }
        tr0[ntx] = nr; te0[ntx] = ne;
        if (nr > 0xfffffff0u || ne > 0xfffffff0u) return -2;
        if (nr > c->nruns) {
            const size_t cap = nr + nr / 4 + 4096; c->nruns = 0;
            if (dgrow(&c->d_runs, nullptr, cap, sizeof(w4d_run)) || hgrow(&c->h_runs, cap, sizeof(w4d_run)) || dgrow(&c->d_lut, nullptr, 2 * cap, 1) || hgrow(&c->h_lut, 2 * cap, 1)) return -2;
            c->nruns = cap;
        }
        if (ne > c->nev) {
            const size_t cap = ne + ne / 4 + 4096; c->nev = 0;
            if (dgrow(&c->d_ev, nullptr, cap, sizeof(w4d_ev)) || hgrow(&c->h_ev, cap, sizeof(w4d_ev))) return -2;
            c->nev = cap;
        }
        CK(cudaMemcpyAsync(c->d_off, c->h_off, 2 * rows * 4, cudaMemcpyHostToDevice, c->st));
        if (nr) {
            w4_events_k<<<rb, 128, 0, c->st>>>(c->d_ph, c->d_R, nz, ny, W, T, ntx, ms, 1, c->d_off, c->d_off + rows, nullptr, c->d_runs, c->d_ev);
            CK(cudaMemcpyAsync(c->h_runs, c->d_runs, nr * sizeof(w4d_run), cudaMemcpyDeviceToHost, c->st));
            if (ne) CK(cudaMemcpyAsync(c->h_ev, c->d_ev, ne * sizeof(w4d_ev), cudaMemcpyDeviceToHost, c->st));
        }
        CK(w4_wait(c->st));
        static int prof = -1; if (prof < 0) prof = getenv("UFSM_W4DEV_PROF") != nullptr;
        struct timespec t0, t1; if (prof) clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
        uint8_t *ro = c->h_lut, *cb = c->h_lut + nr;
        for (int tx = 0; tx < ntx; tx++) {
            const size_t a = tr0[tx], n = tr0[tx + 1] - a;
            const int nc = replay(c->h_runs + a, n, c->h_ev + te0[tx], te0[tx + 1] - te0[tx], (uint32_t)a, minsize, mc, ro + a, cb + a, &acc);
            if (nc < 0) return -1;
            const int base = next; next += nc;
            if (!one) for (size_t k = a; k < a + n; k++) { if (cb[k] == 255 || base + cb[k] > 254) { cb[k] = 255; ro[k] = 255; } else cb[k] = (uint8_t)(base + cb[k]); }
        }
        if (prof) { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1); static double tr; static size_t se, sr; tr += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec); se += ne; sr += nr;
                    if (z0 + T >= D && y0 + T >= H) fprintf(stderr, "band_wind4_dev: replay %.3f CPU-s so far, %zu runs, %zu events\n", tr, sr, se); }
        if (ro_out) {   /* (the runs path: the offsets and LUT kept, no planes) */
            if ((uint64_t)gbase + nr > 0xfffffff0u || runs_append(ro_out, c->h_off, rows, gbase, ro, cb, nr)) return -1;
            gbase += (uint32_t)nr; continue;
        }
        if (nr) CK(cudaMemcpyAsync(c->d_lut, c->h_lut, 2 * nr, cudaMemcpyHostToDevice, c->st));
        w4_out_k<<<(unsigned)((vox + 255) / 256), 256, 0, c->st>>>(c->d_ph, c->d_R, nz, ny, W, T, ntx, c->d_off, c->d_lut, c->d_lut + nr, c->d_c);
        CK(cudaMemcpy2DAsync(w4 + off0, (size_t)H * W, c->d_ph, (size_t)ny * W, (size_t)ny * W, nz, cudaMemcpyDeviceToHost, c->st));
        CK(cudaMemcpy2DAsync(comp + off0, (size_t)H * W, c->d_c, (size_t)ny * W, (size_t)ny * W, nz, cudaMemcpyDeviceToHost, c->st));
        CK(w4_wait(c->st));
        if (cudaGetLastError() != cudaSuccess) return -2;
    }
    if (!one && next > 255) next = 255;
    acc.ncomp = next; if (ro_out) ro_out->ncomp = next;
    if (st) *st = acc;
    return next;
}
struct w4app { uint8_t *d_ph, *d_c; uint16_t *d_R; uint32_t *d_cnt; size_t vox, rows; uint32_t *d_off; size_t noff; uint8_t *d_lut; size_t nlut; };   /* (per device: the trainer uploads on one copy stream per device, so the scratch is reused in stream order) */
w4app g_app[MAXDEV]; pthread_mutex_t g_app_mu[MAXDEV] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
                                                           PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
}   // namespace
extern "C" void w4dev_fail_inject(int n) { g_w4_inject = n; }

extern "C" int w4runs_available(void) { return 1; }
extern "C" int band_wind4_runs(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, size_t minsize, int maxcomp, w4runs *o, band_wind4_stats *st) {
    o->tile = tile; o->ms = maxstep;
    if (g_nctx && w4_dev_ok()) {
        int one, T, ntx; w4_geom(D, H, W, tile, &one, &T, &ntx);
        if (T < 16384) {
            w4ctx *c = nullptr;
            for (int k = 0; k < g_nctx && !c; k++) { w4ctx *t = &g_ctx[(g_next + k) % g_nctx]; if (!pthread_mutex_trylock(&t->mu)) c = t; }
            if (!c) { c = &g_ctx[__sync_fetch_and_add(&g_next, 1) % g_nctx]; pthread_mutex_lock(&c->mu); } else __sync_fetch_and_add(&g_next, 1);
            const int rc = ctx_run(c, phase, D, H, W, tile, maxstep, minsize, maxcomp, nullptr, nullptr, st, o);
            pthread_mutex_unlock(&c->mu);
            if (rc != -2) return rc;
            w4_dev_failed("band_wind4_runs", c->dev);
        }
    }
    return cpu_runs(phase, D, H, W, tile, maxstep, minsize, maxcomp, o, st);
}
extern "C" void w4runs_apply_host(const uint8_t *phase, int D, int H, int W, int tile, int ms, const w4runs *o, uint8_t *w4, uint8_t *comp) {
    int one, T, ntx; w4_geom(D, H, W, tile, &one, &T, &ntx); (void)one;
    const size_t sv = (size_t)(T < D ? T : D) * (T < H ? T : H) * W;
    uint8_t *ph = (uint8_t *)malloc(sv), *cp = (uint8_t *)malloc(sv); uint16_t *R = (uint16_t *)malloc(sv * 2); uint32_t *cnt = (uint32_t *)malloc((sv / W + 1) * ntx * 4);
    if (!ph || !cp || !R || !cnt) abort();
    size_t ob = 0;
    for (int z0 = 0; z0 < D; z0 += T) for (int y0 = 0; y0 < H; y0 += T) {
        const int nz = D - z0 < T ? D - z0 : T, ny = H - y0 < T ? H - y0 : T; const size_t rows = (size_t)nz * ny * ntx, vox = (size_t)nz * ny * W;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) memcpy(ph + ((size_t)z * ny + y) * W, phase + ((size_t)(z0 + z) * H + y0 + y) * W, (size_t)W);
        for (size_t r = 0; r < rows; r++) w4_runs_row(r, ph, nz, ny, W, T, ntx, ms, R, cnt);
        for (size_t i = 0; i < vox; i++) w4_out_vox(i, ph, R, nz, ny, W, T, ntx, o->off + ob, o->ro, o->cb, cp);
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
            const size_t a = ((size_t)z * ny + y) * W, b = ((size_t)(z0 + z) * H + y0 + y) * W;
            memcpy(w4 + b, ph + a, (size_t)W); memcpy(comp + b, cp + a, (size_t)W);
        }
        ob += rows;
    }
    free(ph); free(cp); free(R); free(cnt);
}
extern "C" int nn_w4runs_apply(const uint8_t *phase_d, int D, int H, int W, int tile, int ms, const w4runs *o, uint8_t *w4_d, uint8_t *comp_d, void *stream) {
    int dv = 0; if (cudaGetDevice(&dv) != cudaSuccess) return -1; dv &= MAXDEV - 1;
    cudaStream_t s = (cudaStream_t)stream; int one, T, ntx; w4_geom(D, H, W, tile, &one, &T, &ntx); (void)one;
    const size_t sv = (size_t)(T < D ? T : D) * (T < H ? T : H) * W;
    pthread_mutex_lock(&g_app_mu[dv]); w4app *a = &g_app[dv]; int rc = 0;
#define AK(x) do { if ((x) != cudaSuccess) { rc = -1; goto out; } } while (0)
    if (w4_inj()) { rc = -1; goto out; }
    if (sv > a->vox) { cudaFree(a->d_ph); cudaFree(a->d_c); cudaFree(a->d_R); a->vox = 0; AK(cudaMalloc(&a->d_ph, sv)); AK(cudaMalloc(&a->d_c, sv)); AK(cudaMalloc(&a->d_R, 2 * sv)); a->vox = sv; }
    { const size_t rmax = (size_t)(T < D ? T : D) * (T < H ? T : H) * ntx; if (rmax > a->rows) { cudaFree(a->d_cnt); a->rows = 0; AK(cudaMalloc(&a->d_cnt, rmax * 4)); a->rows = rmax; } }
    if (o->noff > a->noff) { cudaFree(a->d_off); a->noff = 0; AK(cudaMalloc(&a->d_off, (o->noff + o->noff / 4 + 1024) * 4)); a->noff = o->noff + o->noff / 4 + 1024; }
    if (2 * o->nr > a->nlut) { cudaFree(a->d_lut); a->nlut = 0; AK(cudaMalloc(&a->d_lut, 2 * o->nr + o->nr / 2 + 8192)); a->nlut = 2 * o->nr + o->nr / 2 + 8192; }
    AK(cudaMemcpyAsync(a->d_off, o->off, o->noff * 4, cudaMemcpyHostToDevice, s));
    if (o->nr) { AK(cudaMemcpyAsync(a->d_lut, o->ro, o->nr, cudaMemcpyHostToDevice, s)); AK(cudaMemcpyAsync(a->d_lut + o->nr, o->cb, o->nr, cudaMemcpyHostToDevice, s)); }
    {
        size_t ob = 0;
        for (int z0 = 0; z0 < D; z0 += T) for (int y0 = 0; y0 < H; y0 += T) {
            const int nz = D - z0 < T ? D - z0 : T, ny = H - y0 < T ? H - y0 : T; const size_t rows = (size_t)nz * ny * ntx, vox = (size_t)nz * ny * W, off0 = ((size_t)z0 * H + y0) * W;
            AK(cudaMemcpy2DAsync(a->d_ph, (size_t)ny * W, phase_d + off0, (size_t)H * W, (size_t)ny * W, nz, cudaMemcpyDeviceToDevice, s));
            w4_runs_k<<<(unsigned)((rows + 127) / 128), 128, 0, s>>>(a->d_ph, nz, ny, W, T, ntx, ms, a->d_R, a->d_cnt);   /* (the run counts: unused here) */
            w4_out_k<<<(unsigned)((vox + 255) / 256), 256, 0, s>>>(a->d_ph, a->d_R, nz, ny, W, T, ntx, a->d_off + ob, a->d_lut, a->d_lut + o->nr, a->d_c);
            AK(cudaMemcpy2DAsync(w4_d + off0, (size_t)H * W, a->d_ph, (size_t)ny * W, (size_t)ny * W, nz, cudaMemcpyDeviceToDevice, s));
            AK(cudaMemcpy2DAsync(comp_d + off0, (size_t)H * W, a->d_c, (size_t)ny * W, (size_t)ny * W, nz, cudaMemcpyDeviceToDevice, s));
            ob += rows;
        }
    }
    AK(cudaGetLastError());
out:
#undef AK
    if (rc) {   /* (no memory beside the trainer: the scratch freed, the next call allocates it again; the caller takes the host) */
        cudaStreamSynchronize(s); cudaFree(a->d_ph); cudaFree(a->d_c); cudaFree(a->d_R); cudaFree(a->d_cnt); cudaFree(a->d_off); cudaFree(a->d_lut);
        a->d_ph = a->d_c = nullptr; a->d_R = nullptr; a->d_cnt = a->d_off = nullptr; a->d_lut = nullptr; a->vox = a->rows = a->noff = a->nlut = 0;
        cudaGetLastError();
    }
    pthread_mutex_unlock(&g_app_mu[dv]);
    return rc;
}

extern "C" void w4dev_init(const int *devices, int n) {
    g_nctx = 0;
    for (int i = 0; i < n && i < MAXDEV; i++) for (int k = 0; k < NCTX; k++) { w4ctx *c = &g_ctx[g_nctx++]; memset(c, 0, sizeof *c); c->dev = devices[i]; pthread_mutex_init(&c->mu, nullptr); }
    g_bad = 0;
}
extern "C" int band_wind4_dev(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, size_t minsize, int maxcomp, uint8_t *w4, uint8_t *comp, band_wind4_stats *st) {
    if (!g_nctx || !w4_dev_ok()) return -2;
    { const int one = tile <= 0 || (tile >= D && tile >= H && tile >= W), T = one ? (D > H ? (D > W ? D : W) : (H > W ? H : W)) : tile; if (T >= 16384) return -2; }
    w4ctx *c = nullptr;
    for (int k = 0; k < g_nctx && !c; k++) { w4ctx *t = &g_ctx[(g_next + k) % g_nctx]; if (!pthread_mutex_trylock(&t->mu)) c = t; }
    if (!c) { c = &g_ctx[__sync_fetch_and_add(&g_next, 1) % g_nctx]; pthread_mutex_lock(&c->mu); } else __sync_fetch_and_add(&g_next, 1);
    const int rc = ctx_run(c, phase, D, H, W, tile, maxstep, minsize, maxcomp, w4, comp, st);
    pthread_mutex_unlock(&c->mu);
    if (rc == -2) w4_dev_failed("band_wind4_dev", c->dev);
    return rc;
}
