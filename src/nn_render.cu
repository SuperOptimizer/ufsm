/* ufsm render on the GPU (src/render.c render_wraps_chunked_fill with render_set_gpu(1)): a device pool of decoded chunk slots (the
   decoder threads write lossy substreams straight into a slot through gpudec_decode_dev, or upload a host-decoded chunk from a pinned
   staging buffer), a device table of chunk pointers per volume, and one kernel launch per run of ready blocks that evaluates
   render_core.h rc_pixel for every pixel of every grid. Built WITHOUT fast math and with -fmad=false (Makefile): the renders and fill
   flags are the CPU path's bytes. Any device error -> the caller re-renders on the CPU. Called through weak symbols (render.c): a
   binary without this object (test_render_stream / test_render_fill CPU builds) is CPU-only. */
#include "render_core.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { RB = 32, SEGSLOTS = 64, MAXSTG = 512, MAXSEG = 4096 };   /* staging: two per decoder thread (render runs up to 128 + 128) */

typedef struct { uint8_t *h; cudaStream_t st; cudaEvent_t ev; } rg_stage;
typedef struct { uint8_t *seg[MAXSEG]; int nseg; uint8_t **freel; size_t nfree, capfree, cap, used; } rg_tier;   /* a slot pool: segments, free slots, bytes */
struct rgpu {
    int dev, broken; size_t cv;                            /* slot bytes */
    pthread_mutex_t mu; rg_tier tier[2];                   /* device slots, mapped pinned host slots */
    rg_stage stg[MAXSTG]; int nstg;
    cudaStream_t st; cudaEvent_t e0[2], e1[2];             /* per launch buffer: timing; e1 blocking-sync (the driver sleeps, it does not spin) */
    /* the render */
    int nv, ns, w, h, nimg; const uint8_t **dptr[2]; size_t nc[2];
    float *X[8], *Y[8], *Z[8]; uint8_t *vg[8]; int32_t *pxy; size_t nb, bmax;
    uint8_t *dout[2], *hout[2]; size_t outb;               /* two launch buffers (mapped pinned) */
    uint64_t *upd_d[2], *upd_h[2]; size_t ucap[2];         /* pointer updates: (v << 63 | ci, ptr) pairs */
    double kms; uint64_t nlaunch, nput, bytes;
};

static int rg_err(rgpu *G, cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 0;
    cudaGetLastError(); G->broken = 1; fprintf(stderr, "render gpu: %s: %s: CPU render\n", what, cudaGetErrorString(e)); return -1;
}
extern "C" size_t rgpu_free_mem(int dev, size_t *total) {
    size_t fr = 0, tot = 0;
    if (cudaSetDevice(dev) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (total) *total = tot;
    return fr;
}
extern "C" void rgpu_close(rgpu *G);
/* dev: the CUDA device; cv: slot bytes (a chunk); cap / hcap: device / host slot bytes at most; nstg pinned staging buffers of cv bytes */
extern "C" rgpu *rgpu_open(int dev, size_t cv, size_t cap, size_t hcap, int nstg) {
    int nd = 0; if (cudaGetDeviceCount(&nd) != cudaSuccess || nd <= dev) { cudaGetLastError(); return nullptr; }
    rgpu *G = (rgpu *)calloc(1, sizeof *G); if (!G) return nullptr;
    G->dev = dev; G->cv = cv; G->tier[0].cap = cap; G->tier[1].cap = hcap; G->nstg = nstg > MAXSTG ? MAXSTG : nstg;
    pthread_mutex_init(&G->mu, nullptr);
    int lo = 0, hi = 0; cudaDeviceGetStreamPriorityRange(&lo, &hi);   /* the render stream ahead of the decoder's batches */
    int ok = cudaSetDevice(dev) == cudaSuccess && cudaStreamCreateWithPriority(&G->st, cudaStreamNonBlocking, hi) == cudaSuccess &&
             cudaEventCreate(&G->e0[0]) == cudaSuccess && cudaEventCreate(&G->e0[1]) == cudaSuccess &&
             cudaEventCreateWithFlags(&G->e1[0], cudaEventBlockingSync) == cudaSuccess && cudaEventCreateWithFlags(&G->e1[1], cudaEventBlockingSync) == cudaSuccess;
    for (int k = 0; k < G->nstg && ok; k++)
        ok = cudaHostAlloc((void **)&G->stg[k].h, cv, cudaHostAllocDefault) == cudaSuccess && cudaStreamCreateWithFlags(&G->stg[k].st, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&G->stg[k].ev, cudaEventDisableTiming | cudaEventBlockingSync) == cudaSuccess;
    if (!ok) { cudaGetLastError(); rgpu_close(G); return nullptr; }
    return G;
}
extern "C" void rgpu_close(rgpu *G) {
    if (!G) return;
    cudaSetDevice(G->dev);
    if (G->st) cudaStreamSynchronize(G->st);
    for (int k = 0; k < G->nstg; k++) { if (G->stg[k].st) { cudaStreamSynchronize(G->stg[k].st); cudaStreamDestroy(G->stg[k].st); } if (G->stg[k].ev) cudaEventDestroy(G->stg[k].ev); if (G->stg[k].h) cudaFreeHost(G->stg[k].h); }
    for (int k = 0; k < G->tier[0].nseg; k++) cudaFree(G->tier[0].seg[k]);
    for (int k = 0; k < G->tier[1].nseg; k++) cudaFreeHost(G->tier[1].seg[k]);
    for (int v = 0; v < 2; v++) cudaFree(G->dptr[v]);
    for (int s = 0; s < 8; s++) { cudaFree(G->X[s]); cudaFree(G->Y[s]); cudaFree(G->Z[s]); cudaFree(G->vg[s]); }
    cudaFree(G->pxy);
    for (int b = 0; b < 2; b++) { if (G->e0[b]) cudaEventDestroy(G->e0[b]); if (G->e1[b]) cudaEventDestroy(G->e1[b]); if (G->hout[b]) cudaFreeHost(G->hout[b]); if (G->upd_h[b]) cudaFreeHost(G->upd_h[b]); }
    if (G->st) cudaStreamDestroy(G->st);
    cudaGetLastError();
    free(G->tier[0].freel); free(G->tier[1].freel); pthread_mutex_destroy(&G->mu); free(G);
}
extern "C" int rgpu_broken(const rgpu *G) { return !G || G->broken; }
extern "C" uint8_t *rgpu_stage(rgpu *G, int k) { return k >= 0 && k < G->nstg ? G->stg[k].h : nullptr; }
extern "C" void rgpu_stats(const rgpu *G, size_t *pool, size_t *hpool, double *kms, uint64_t *nlaunch, uint64_t *nput, uint64_t *bytes) {
    *pool = G->tier[0].used; *hpool = G->tier[1].used; *kms = G->kms; *nlaunch = G->nlaunch; *nput = G->nput; *bytes = G->bytes;
}
/* a free slot. Tier 0: device memory (segments of SEGSLOTS slots allocated while under the cap); tier 1 (when the device tier is full,
   e.g. a P3 stripe's live chunk set > the GPU): mapped pinned host memory the kernel reads over the bus (the decoders write it directly).
   *host: the tier. nullptr: both full or a device failure */
static uint8_t *rg_take(rgpu *G, rg_tier *T, int host) {   /* under G->mu */
    if (!T->nfree) {
        const size_t ns = host || T->cap >= G->cv * SEGSLOTS ? SEGSLOTS : T->cap / G->cv, sb = G->cv * ns; uint8_t *p = nullptr;   /* a small cap (tests): one smaller segment */
        if (!ns || T->nseg == MAXSEG || T->used + sb > T->cap || cudaSetDevice(G->dev) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        if (host) {
            void *dp = nullptr;
            if (cudaHostAlloc((void **)&p, sb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return nullptr; }
            if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess || dp != (void *)p) { cudaGetLastError(); cudaFreeHost(p); T->cap = 0; return nullptr; }   /* needs UVA */
        } else if (cudaMalloc(&p, sb) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        if (T->capfree < T->nfree + ns) { T->capfree = T->nfree + ns + T->capfree; uint8_t **q = (uint8_t **)realloc(T->freel, T->capfree * sizeof *q);
                                                if (!q) { if (host) cudaFreeHost(p); else cudaFree(p); return nullptr; } T->freel = q; }
        T->seg[T->nseg++] = p; T->used += sb;
        for (size_t k = ns; k-- > 0;) T->freel[T->nfree++] = p + k * G->cv;
    }
    return T->freel[--T->nfree];
}
extern "C" uint8_t *rgpu_slot(rgpu *G, int *host) {
    if (G->broken) return nullptr;
    pthread_mutex_lock(&G->mu);
    uint8_t *d = rg_take(G, &G->tier[0], 0); *host = 0;
    if (!d && (d = rg_take(G, &G->tier[1], 1))) *host = 1;
    pthread_mutex_unlock(&G->mu);
    return d;
}
static int rg_in(const rg_tier *T, const uint8_t *d, size_t sb) { for (int k = 0; k < T->nseg; k++) if (d >= T->seg[k] && d < T->seg[k] + sb) return 1; return 0; }
extern "C" void rgpu_release(rgpu *G, uint8_t *d) {
    pthread_mutex_lock(&G->mu);
    rg_tier *T = &G->tier[rg_in(&G->tier[1], d, G->cv * SEGSLOTS)];
    if (T->nfree == T->capfree) { T->capfree = T->capfree ? 2 * T->capfree : 256; T->freel = (uint8_t **)realloc(T->freel, T->capfree * sizeof *T->freel); }
    T->freel[T->nfree++] = d;
    pthread_mutex_unlock(&G->mu);
}
/* slot d := 0xa5 (UFSM_RENDER_GPU_POISON, tests: substreams nobody asked for must never be read); synchronous */
extern "C" int rgpu_zero(rgpu *G, int k, uint8_t *d) {
    if (G->broken) return -1;
    cudaError_t e = cudaSetDevice(G->dev);
    if (e == cudaSuccess) e = cudaMemsetAsync(d, 0xa5, G->cv, G->stg[k].st);
    if (e == cudaSuccess) e = cudaEventRecord(G->stg[k].ev, G->stg[k].st);
    if (e == cudaSuccess) e = cudaEventSynchronize(G->stg[k].ev);
    return rg_err(G, e, "slot memset");
}
/* staging buffer k (a host-decoded chunk, 128^3 layout when sub) -> slot d: the substreams in mask (bit s: z 16 (s / 4).., y 32 (s % 4)..)
   or the whole chunk (mask 0 / all ones); box 1: the substreams' z x y bounding box in one copy (the host buffer's other substreams are 0
   and never read; not when the device holds some); wait 0: asynchronous, rgpu_stage_sync(G, k) before the buffer or the slot is used again */
extern "C" int rgpu_upload(rgpu *G, int k, uint8_t *d, uint32_t mask, int box, int wait) {
    if (G->broken) return -1;
    cudaError_t e = cudaSetDevice(G->dev); const uint8_t *h = G->stg[k].h; const cudaStream_t st = G->stg[k].st; size_t bytes = 0;
    if (e == cudaSuccess && (!mask || mask == 0xffffffffu)) { e = cudaMemcpyAsync(d, h, G->cv, cudaMemcpyHostToDevice, st); bytes = G->cv; }
    else if (e == cudaSuccess && box) {
        int z0 = 8, z1 = -1, y0 = 4, y1 = -1;
        for (int b = 0; b < 32; b++) if (mask >> b & 1) { const int bz = b / 4, by = b % 4; z0 = bz < z0 ? bz : z0; z1 = bz > z1 ? bz : z1; y0 = by < y0 ? by : y0; y1 = by > y1 ? by : y1; }
        const size_t o = ((size_t)16 * z0 * 128 + 32 * y0) * 128, wb = (size_t)(y1 - y0 + 1) * 32 * 128, nz = (size_t)(z1 - z0 + 1) * 16;
        e = wb == (size_t)128 * 128 ? cudaMemcpyAsync(d + o, h + o, wb * nz, cudaMemcpyHostToDevice, st) : cudaMemcpy2DAsync(d + o, (size_t)128 * 128, h + o, (size_t)128 * 128, wb, nz, cudaMemcpyHostToDevice, st);
        bytes = wb * nz;
    }
    else for (int b = 0; b < 32 && e == cudaSuccess; b++) if (mask >> b & 1) {
        const size_t o = ((size_t)16 * (b / 4) * 128 + 32 * (b % 4)) * 128;
        e = cudaMemcpy2DAsync(d + o, (size_t)128 * 128, h + o, (size_t)128 * 128, 32 * 128, 16, cudaMemcpyHostToDevice, st); bytes += 16 * 32 * 128;
    }
    if (e == cudaSuccess) e = cudaEventRecord(G->stg[k].ev, st);
    if (e == cudaSuccess && wait) e = cudaEventSynchronize(G->stg[k].ev);
    if (rg_err(G, e, "chunk upload")) return -1;
    __atomic_fetch_add(&G->nput, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&G->bytes, bytes, __ATOMIC_RELAXED);
    return 0;
}
/* the render: nv volumes with nc[v] chunks in their tables, ns grids (w x h; vgrid[s] or nullptr), the blocks' (px, py) in render order,
   at most bmax blocks per launch, nimg output bytes per pixel (nv, + 1 with the fill flags) */
extern "C" int rgpu_setup(rgpu *G, int nv, const size_t nc[2], int ns, const float *const *X, const float *const *Y, const float *const *Z, int w, int h,
                          const uint8_t *const *vgrid, const int32_t *pxy, size_t nb, size_t bmax, int nimg) {
    if (G->broken || ns > 8 || nv > 2) return -1;
    cudaError_t e = cudaSetDevice(G->dev); const size_t n = (size_t)w * h;
    G->nv = nv; G->ns = ns; G->w = w; G->h = h; G->nb = nb; G->bmax = bmax; G->nimg = nimg;
    for (int v = 0; v < nv && e == cudaSuccess; v++) {
        G->nc[v] = nc[v];
        e = cudaMalloc(&G->dptr[v], nc[v] * sizeof(uint8_t *));
        if (e == cudaSuccess) e = cudaMemsetAsync(G->dptr[v], 0, nc[v] * sizeof(uint8_t *), G->st);
    }
    for (int s = 0; s < ns && e == cudaSuccess; s++) {
        const float *src[3] = {X[s], Y[s], Z[s]}; float **dst[3] = {&G->X[s], &G->Y[s], &G->Z[s]};
        for (int d = 0; d < 3 && e == cudaSuccess; d++) { e = cudaMalloc(dst[d], n * sizeof(float)); if (e == cudaSuccess) e = cudaMemcpy(*dst[d], src[d], n * sizeof(float), cudaMemcpyHostToDevice); }
        if (vgrid && vgrid[s] && e == cudaSuccess) { e = cudaMalloc(&G->vg[s], n); if (e == cudaSuccess) e = cudaMemcpy(G->vg[s], vgrid[s], n, cudaMemcpyHostToDevice); }
    }
    if (e == cudaSuccess && nb) { e = cudaMalloc(&G->pxy, nb * 2 * sizeof(int32_t)); if (e == cudaSuccess) e = cudaMemcpy(G->pxy, pxy, nb * 2 * sizeof(int32_t), cudaMemcpyHostToDevice); }
    G->outb = bmax * ns * nimg * RB * RB;
    /* two launches in flight (the driver scatters one while the next renders); the kernel writes its outputs straight into mapped pinned
       host memory and reads the pointer updates from there: the driver never queues behind the decoders' uploads on the copy engine */
    for (int b = 0; b < 2 && e == cudaSuccess; b++) {
        e = cudaHostAlloc((void **)&G->hout[b], G->outb, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e == cudaSuccess) e = cudaHostGetDevicePointer((void **)&G->dout[b], G->hout[b], 0);
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(G->st);
    return rg_err(G, e, "setup");
}
__global__ void k_ptrs(const uint8_t **p0, const uint8_t **p1, const uint64_t *u, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const uint64_t key = u[2 * i]; const uint8_t *p = (const uint8_t *)u[2 * i + 1];
    (key >> 63 ? p1 : p0)[key & ~(1ull << 63)] = p;
}
/* chunk keys[i] (ci | v << 63) is now at device address ptrs[i] (nullptr: zero), before launch b (buffer b free: its last launch waited) */
extern "C" int rgpu_ptrs(rgpu *G, int b, const uint64_t *keys, uint8_t *const *ptrs, size_t n) {
    if (G->broken) return -1;
    if (!n) return 0;
    cudaError_t e = cudaSetDevice(G->dev);
    if (e == cudaSuccess && n > G->ucap[b]) {
        if (G->upd_h[b]) cudaFreeHost(G->upd_h[b]);
        G->upd_d[b] = nullptr; G->upd_h[b] = nullptr; G->ucap[b] = 0;
        const size_t nn = n < 4096 ? 4096 : 2 * n;
        e = cudaHostAlloc((void **)&G->upd_h[b], 2 * nn * sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable);
        if (e == cudaSuccess) e = cudaHostGetDevicePointer((void **)&G->upd_d[b], G->upd_h[b], 0);
        if (e == cudaSuccess) G->ucap[b] = nn;
    }
    if (e != cudaSuccess) return rg_err(G, e, "pointer table");
    for (size_t i = 0; i < n; i++) { G->upd_h[b][2 * i] = keys[i]; G->upd_h[b][2 * i + 1] = (uint64_t)(uintptr_t)ptrs[i]; }
    k_ptrs<<<(unsigned)((n + 255) / 256), 256, 0, G->st>>>(G->dptr[0], G->dptr[1], G->upd_d[b], n);
    return rg_err(G, cudaGetLastError(), "pointer table");
}

typedef struct { rc_vol V[2]; int depth[2]; rc_fill F; int fill, nv, ns, w, h, ow, oh, nimg; double gs; const float *X[8], *Y[8], *Z[8]; const uint8_t *vg[8]; } rg_par;
/* block (kb + blockIdx.x), grid blockIdx.y: 32 x 8 threads, 4 rows each; out[((k ns + s) nimg + q) 1024 + r 32 + c]: volume q's value (q < nv),
   the fill flag (q = nv); pixels without a surface (or outside the image): 0 / RFILL_NONE (254), what the caller's images start as */
__global__ void __launch_bounds__(256) k_render(rg_par P, const int32_t *pxy, size_t kb, uint8_t *out) {
    const size_t k = blockIdx.x; const int s = blockIdx.y, c = threadIdx.x;
    const int px = pxy[2 * (kb + k)] + c, py0 = pxy[2 * (kb + k) + 1];
    uint8_t *o = out + ((k * P.ns + s) * P.nimg) * (RB * RB);
    for (int r = threadIdx.y; r < RB; r += 8) {
        const int py = py0 + r; uint8_t val[2] = {0, 0}, fl = 254;
        if (px < P.ow && py < P.oh) rc_pixel(P.Z[s], P.Y[s], P.X[s], P.w, P.h, P.gs, px, py, P.V, P.nv, P.depth, P.fill ? &P.F : nullptr, P.vg[s], val, &fl);
        for (int q = 0; q < P.nv; q++) o[q * RB * RB + r * RB + c] = val[q];
        if (P.fill) o[P.nv * RB * RB + r * RB + c] = fl;
    }
}
/* blocks [kb, ke) rendered into buffer b, asynchronously (rgpu_wait(G, b) for the outputs) */
extern "C" int rgpu_launch(rgpu *G, int b, size_t kb, size_t ke, const rc_vol *V, const int *depth, const rc_fill *F, double gs, int ow, int oh) {
    if (G->broken) return -1;
    if (ke <= kb || ke - kb > G->bmax) return -1;
    { const char *fe = getenv("UFSM_RENDER_GPU_FAIL"); if (fe && (uint64_t)atoi(fe) == G->nlaunch + 1) return rg_err(G, cudaErrorUnknown, "injected failure (UFSM_RENDER_GPU_FAIL, tests)"); }
    rg_par P; memset(&P, 0, sizeof P);
    for (int v = 0; v < G->nv; v++) { P.V[v] = V[v]; P.V[v].ptr = G->dptr[v]; P.depth[v] = depth[v]; }
    if (F) { P.F = *F; P.fill = 1; }
    P.nv = G->nv; P.ns = G->ns; P.w = G->w; P.h = G->h; P.ow = ow; P.oh = oh; P.nimg = G->nimg; P.gs = gs;
    for (int s = 0; s < G->ns; s++) { P.X[s] = G->X[s]; P.Y[s] = G->Y[s]; P.Z[s] = G->Z[s]; P.vg[s] = G->vg[s]; }
    cudaError_t e = cudaSetDevice(G->dev);
    if (e == cudaSuccess) e = cudaEventRecord(G->e0[b], G->st);
    if (e == cudaSuccess) { k_render<<<dim3((unsigned)(ke - kb), (unsigned)G->ns), dim3(32, 8), 0, G->st>>>(P, G->pxy, kb, G->dout[b]); e = cudaGetLastError(); }
    if (e == cudaSuccess) e = cudaEventRecord(G->e1[b], G->st);
    G->nlaunch++;
    return rg_err(G, e, "render kernel");
}
/* launch b done; *hout: its outputs (layout of k_render, k relative to its kb) */
extern "C" int rgpu_wait(rgpu *G, int b, const uint8_t **hout) {
    if (G->broken) return -1;
    cudaError_t e = cudaEventSynchronize(G->e1[b]);
    float ms = 0; if (e == cudaSuccess && cudaEventElapsedTime(&ms, G->e0[b], G->e1[b]) == cudaSuccess) G->kms += ms;
    *hout = G->hout[b];
    return rg_err(G, e, "render kernel");
}
/* the last asynchronous upload from staging buffer k is done */
extern "C" int rgpu_stage_sync(rgpu *G, int k) {
    if (G->broken) return -1;
    return rg_err(G, cudaEventSynchronize(G->stg[k].ev), "chunk upload");
}
