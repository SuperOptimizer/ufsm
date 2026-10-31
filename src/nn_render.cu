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

enum { RB = 32, SEGSLOTS = 64, MAXSTG = 192, MAXSEG = 4096 };

typedef struct { uint8_t *h; cudaStream_t st; } rg_stage;
struct rgpu {
    int dev, broken; size_t cv, cap, used;                 /* slot bytes; pool cap and bytes allocated */
    pthread_mutex_t mu; uint8_t *seg[MAXSEG]; int nseg; uint8_t **freel; size_t nfree, capfree;
    rg_stage stg[MAXSTG]; int nstg;
    cudaStream_t st;
    /* the render */
    int nv, ns, w, h, nimg; const uint8_t **dptr[2]; size_t nc[2];
    float *X[8], *Y[8], *Z[8]; uint8_t *vg[8]; int32_t *pxy; size_t nb, bmax;
    uint8_t *dout, *hout; size_t outb;
    uint64_t *upd_d, *upd_h; size_t ucap;                  /* pointer updates: (v << 63 | ci, ptr) pairs */
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
/* dev: the CUDA device; cv: slot bytes (a chunk); cap: pool bytes at most; nstg pinned staging buffers of cv bytes (one per decoder) */
extern "C" rgpu *rgpu_open(int dev, size_t cv, size_t cap, int nstg) {
    int nd = 0; if (cudaGetDeviceCount(&nd) != cudaSuccess || nd <= dev) { cudaGetLastError(); return nullptr; }
    rgpu *G = (rgpu *)calloc(1, sizeof *G); if (!G) return nullptr;
    G->dev = dev; G->cv = cv; G->cap = cap; G->nstg = nstg > MAXSTG ? MAXSTG : nstg;
    pthread_mutex_init(&G->mu, nullptr);
    int ok = cudaSetDevice(dev) == cudaSuccess && cudaStreamCreateWithFlags(&G->st, cudaStreamNonBlocking) == cudaSuccess;
    for (int k = 0; k < G->nstg && ok; k++)
        ok = cudaHostAlloc((void **)&G->stg[k].h, cv, cudaHostAllocDefault) == cudaSuccess && cudaStreamCreateWithFlags(&G->stg[k].st, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) { cudaGetLastError(); rgpu_close(G); return nullptr; }
    return G;
}
extern "C" void rgpu_close(rgpu *G) {
    if (!G) return;
    cudaSetDevice(G->dev);
    if (G->st) cudaStreamSynchronize(G->st);
    for (int k = 0; k < G->nstg; k++) { if (G->stg[k].st) { cudaStreamSynchronize(G->stg[k].st); cudaStreamDestroy(G->stg[k].st); } if (G->stg[k].h) cudaFreeHost(G->stg[k].h); }
    for (int k = 0; k < G->nseg; k++) cudaFree(G->seg[k]);
    for (int v = 0; v < 2; v++) cudaFree(G->dptr[v]);
    for (int s = 0; s < 8; s++) { cudaFree(G->X[s]); cudaFree(G->Y[s]); cudaFree(G->Z[s]); cudaFree(G->vg[s]); }
    cudaFree(G->pxy); cudaFree(G->dout); cudaFree(G->upd_d); if (G->hout) cudaFreeHost(G->hout); if (G->upd_h) cudaFreeHost(G->upd_h);
    if (G->st) cudaStreamDestroy(G->st);
    cudaGetLastError();
    free(G->freel); pthread_mutex_destroy(&G->mu); free(G);
}
extern "C" int rgpu_broken(const rgpu *G) { return !G || G->broken; }
extern "C" uint8_t *rgpu_stage(rgpu *G, int k) { return k >= 0 && k < G->nstg ? G->stg[k].h : nullptr; }
extern "C" void rgpu_stats(const rgpu *G, size_t *pool, double *kms, uint64_t *nlaunch, uint64_t *nput, uint64_t *bytes) {
    *pool = G->used; *kms = G->kms; *nlaunch = G->nlaunch; *nput = G->nput; *bytes = G->bytes;
}
/* a free slot (a segment of SEGSLOTS slots allocated when none is free, up to the cap); nullptr: the pool is full or the device failed */
extern "C" uint8_t *rgpu_slot(rgpu *G) {
    if (G->broken) return nullptr;
    pthread_mutex_lock(&G->mu);
    if (!G->nfree) {
        const size_t sb = G->cv * SEGSLOTS; uint8_t *p = nullptr;
        if (G->nseg == MAXSEG || G->used + sb > G->cap || cudaSetDevice(G->dev) != cudaSuccess || cudaMalloc(&p, sb) != cudaSuccess) {
            cudaGetLastError(); pthread_mutex_unlock(&G->mu); return nullptr; }
        if (G->capfree < G->nfree + SEGSLOTS) { G->capfree = G->nfree + SEGSLOTS + G->capfree; uint8_t **q = (uint8_t **)realloc(G->freel, G->capfree * sizeof *q); if (!q) { cudaFree(p); pthread_mutex_unlock(&G->mu); return nullptr; } G->freel = q; }
        G->seg[G->nseg++] = p; G->used += sb;
        for (int k = SEGSLOTS - 1; k >= 0; k--) G->freel[G->nfree++] = p + (size_t)k * G->cv;
    }
    uint8_t *d = G->freel[--G->nfree];
    pthread_mutex_unlock(&G->mu);
    return d;
}
extern "C" void rgpu_release(rgpu *G, uint8_t *d) {
    pthread_mutex_lock(&G->mu);
    if (G->nfree == G->capfree) { G->capfree = G->capfree ? 2 * G->capfree : 256; G->freel = (uint8_t **)realloc(G->freel, G->capfree * sizeof *G->freel); }
    G->freel[G->nfree++] = d;
    pthread_mutex_unlock(&G->mu);
}
/* slot d := 0 (before a partial decode: the substreams nobody asked for read 0, as in the host buffer); synchronous */
extern "C" int rgpu_zero(rgpu *G, int k, uint8_t *d) {
    if (G->broken) return -1;
    cudaError_t e = cudaSetDevice(G->dev);
    if (e == cudaSuccess) e = cudaMemsetAsync(d, 0, G->cv, G->stg[k].st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(G->stg[k].st);
    return rg_err(G, e, "slot memset");
}
/* staging buffer k (a host-decoded chunk, 128^3 layout when sub) -> slot d: the substreams in mask (bit s: z 16 (s / 4).., y 32 (s % 4)..)
   or the whole chunk (mask 0 / all ones); synchronous */
extern "C" int rgpu_upload(rgpu *G, int k, uint8_t *d, uint32_t mask) {
    if (G->broken) return -1;
    cudaError_t e = cudaSetDevice(G->dev); const uint8_t *h = G->stg[k].h; const cudaStream_t st = G->stg[k].st; size_t bytes = 0;
    if (e == cudaSuccess && (!mask || mask == 0xffffffffu)) { e = cudaMemcpyAsync(d, h, G->cv, cudaMemcpyHostToDevice, st); bytes = G->cv; }
    else for (int b = 0; b < 32 && e == cudaSuccess; b++) if (mask >> b & 1) {
        const size_t o = ((size_t)16 * (b / 4) * 128 + 32 * (b % 4)) * 128;
        e = cudaMemcpy2DAsync(d + o, (size_t)128 * 128, h + o, (size_t)128 * 128, 32 * 128, 16, cudaMemcpyHostToDevice, st); bytes += 16 * 32 * 128;
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(st);
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
    if (e == cudaSuccess) e = cudaMalloc(&G->dout, G->outb);
    if (e == cudaSuccess) e = cudaHostAlloc((void **)&G->hout, G->outb, cudaHostAllocDefault);
    if (e == cudaSuccess) e = cudaStreamSynchronize(G->st);
    return rg_err(G, e, "setup");
}
__global__ void k_ptrs(const uint8_t **p0, const uint8_t **p1, const uint64_t *u, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const uint64_t key = u[2 * i]; const uint8_t *p = (const uint8_t *)u[2 * i + 1];
    (key >> 63 ? p1 : p0)[key & ~(1ull << 63)] = p;
}
/* chunk ci of volume v is now at device address p (nullptr: zero) */
extern "C" int rgpu_ptrs(rgpu *G, const uint64_t *keys, uint8_t *const *ptrs, size_t n) {
    if (G->broken) return -1;
    if (!n) return 0;
    cudaError_t e = cudaSetDevice(G->dev);
    if (e == cudaSuccess && n > G->ucap) {
        cudaFree(G->upd_d); if (G->upd_h) cudaFreeHost(G->upd_h); G->upd_d = nullptr; G->upd_h = nullptr; G->ucap = 0;
        e = cudaMalloc(&G->upd_d, 2 * n * sizeof(uint64_t)); if (e == cudaSuccess) e = cudaHostAlloc((void **)&G->upd_h, 2 * n * sizeof(uint64_t), cudaHostAllocDefault);
        if (e == cudaSuccess) G->ucap = n;
    }
    if (e != cudaSuccess) return rg_err(G, e, "pointer table");
    for (size_t i = 0; i < n; i++) { G->upd_h[2 * i] = keys[i]; G->upd_h[2 * i + 1] = (uint64_t)(uintptr_t)ptrs[i]; }
    e = cudaMemcpyAsync(G->upd_d, G->upd_h, 2 * n * sizeof(uint64_t), cudaMemcpyHostToDevice, G->st);
    if (e == cudaSuccess) { k_ptrs<<<(unsigned)((n + 255) / 256), 256, 0, G->st>>>(G->dptr[0], G->dptr[1], G->upd_d, n); e = cudaGetLastError(); }
    return rg_err(G, e, "pointer table");
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
/* blocks [kb, ke) rendered; *hout: the pinned host copy of their outputs (layout of k_render, k relative to kb). Synchronous. */
extern "C" int rgpu_render(rgpu *G, size_t kb, size_t ke, const rc_vol *V, const int *depth, const rc_fill *F, double gs, int ow, int oh, const uint8_t **hout) {
    if (G->broken) return -1;
    if (ke <= kb) { *hout = G->hout; return 0; }
    if (ke - kb > G->bmax) return -1;
    { const char *fe = getenv("UFSM_RENDER_GPU_FAIL"); if (fe && (uint64_t)atoi(fe) == G->nlaunch + 1) return rg_err(G, cudaErrorUnknown, "injected failure (UFSM_RENDER_GPU_FAIL, tests)"); }
    rg_par P; memset(&P, 0, sizeof P);
    for (int v = 0; v < G->nv; v++) { P.V[v] = V[v]; P.V[v].ptr = G->dptr[v]; P.depth[v] = depth[v]; }
    if (F) { P.F = *F; P.fill = 1; }
    P.nv = G->nv; P.ns = G->ns; P.w = G->w; P.h = G->h; P.ow = ow; P.oh = oh; P.nimg = G->nimg; P.gs = gs;
    for (int s = 0; s < G->ns; s++) { P.X[s] = G->X[s]; P.Y[s] = G->Y[s]; P.Z[s] = G->Z[s]; P.vg[s] = G->vg[s]; }
    cudaError_t e = cudaSetDevice(G->dev);
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (e == cudaSuccess) e = cudaEventCreate(&e0);
    if (e == cudaSuccess) e = cudaEventCreate(&e1);
    if (e == cudaSuccess) e = cudaEventRecord(e0, G->st);
    if (e == cudaSuccess) { k_render<<<dim3((unsigned)(ke - kb), (unsigned)G->ns), dim3(32, 8), 0, G->st>>>(P, G->pxy, kb, G->dout); e = cudaGetLastError(); }
    if (e == cudaSuccess) e = cudaEventRecord(e1, G->st);
    const size_t nb = (ke - kb) * G->ns * G->nimg * RB * RB;
    if (e == cudaSuccess) e = cudaMemcpyAsync(G->hout, G->dout, nb, cudaMemcpyDeviceToHost, G->st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(G->st);
    float ms = 0; if (e == cudaSuccess && cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess) G->kms += ms;
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    G->nlaunch++;
    *hout = G->hout;
    return rg_err(G, e, "render kernel");
}
