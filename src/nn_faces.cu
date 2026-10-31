/* ufsm faces on the GPU (src/faces.c faces_wrap, src/faces_cli.c chunk cache): the normal profiles sampled from a device mirror of the
   decoded CT chunks, cut into bands (faces_core.h fc_bands_core), the exact volcomp substreams those profiles read (the decode list),
   and the windowed medians of the consistency passes. The same arithmetic in the same order as the CPU path: this file is built
   WITHOUT fast math and with -fmad=false (Makefile), so results are bit-identical. Called through weak symbols: a binary without this
   object (test_faces) is CPU-only. */
#include "faces_core.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FK = 4, SUB = 4, CS = 128, SEG = 128, NSTG = 16 };   /* bands per vertex, kept-profile stride, chunk edge, slots per device segment
                                                               (256 MB), pinned upload staging buffers */
static const size_t CV = (size_t)CS * CS * CS;

typedef struct { pthread_mutex_t mu; uint8_t *h; cudaStream_t st; cudaEvent_t ev; } fg_stage;
struct fgpu {
    int64_t shape[3]; int g[3]; size_t ng;
    int *ptab, *ptab_d; uint32_t *dhave; int dirty;          /* chunk -> device slot (-1 none), its uploaded substreams */
    uint8_t **seg, **seg_d; int nseg, cseg;                 /* slot s -> seg[s / SEG] + (s % SEG) * CV */
    int *freel; int nfree, capfree, broken;
    pthread_mutex_t mu; fg_stage stg[NSTG]; int stg_next;   /* mu: slots / ptab / dhave / stg_next */
    cudaStream_t st;
    size_t n, ncap; int w, h; float *X, *Y, *Z, *N;          /* the wrap's grid */
    size_t bcap; fc_band *cand; uint8_t *nc; float *ct0; uint8_t *prof; size_t pcap;
    float *min, *mout; size_t mcap;
    uint32_t *need_d;                                       /* fgpu_mark: per chunk the substreams the band's profiles read */
    size_t nput, bytes;
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "faces gpu: %s: %s\n", #x, cudaGetErrorString(e_)); return -1; } } while (0)

extern "C" void fgpu_close(fgpu *G);
extern "C" fgpu *fgpu_open(const int64_t shape[3]) {
    int nd = 0; if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) { cudaGetLastError(); return nullptr; }
    fgpu *G = (fgpu *)calloc(1, sizeof *G); if (!G) return nullptr;
    for (int d = 0; d < 3; d++) { G->shape[d] = shape[d]; G->g[d] = (int)((shape[d] + CS - 1) / CS); }
    G->ng = (size_t)G->g[0] * G->g[1] * G->g[2];
    G->ptab = (int *)malloc(G->ng * sizeof(int)); G->dhave = (uint32_t *)calloc(G->ng, sizeof(uint32_t));
    G->cseg = 256; G->seg = (uint8_t **)calloc(G->cseg, sizeof *G->seg);
    pthread_mutex_init(&G->mu, nullptr);
    int ok = G->ptab && G->dhave && G->seg && cudaStreamCreateWithFlags(&G->st, cudaStreamNonBlocking) == cudaSuccess &&
             cudaMalloc(&G->ptab_d, G->ng * sizeof(int)) == cudaSuccess && cudaMalloc(&G->seg_d, G->cseg * sizeof *G->seg_d) == cudaSuccess &&
             cudaMalloc(&G->need_d, G->ng * sizeof(uint32_t)) == cudaSuccess && cudaMemset(G->need_d, 0, G->ng * sizeof(uint32_t)) == cudaSuccess;
    for (int k = 0; k < NSTG && ok; k++) {
        fg_stage *S = &G->stg[k]; pthread_mutex_init(&S->mu, nullptr);
        ok = cudaHostAlloc((void **)&S->h, CV, cudaHostAllocDefault) == cudaSuccess && cudaStreamCreateWithFlags(&S->st, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&S->ev, cudaEventDisableTiming) == cudaSuccess && cudaEventRecord(S->ev, S->st) == cudaSuccess;
    }
    if (!ok) { cudaGetLastError(); fgpu_close(G); return nullptr; }
    memset(G->ptab, 0xff, G->ng * sizeof(int)); G->dirty = 1;
    return G;
}
extern "C" void fgpu_close(fgpu *G) {
    if (!G) return;
    if (G->st) cudaStreamSynchronize(G->st);
    for (int k = 0; k < NSTG; k++) { fg_stage *S = &G->stg[k]; if (S->st) { cudaStreamSynchronize(S->st); cudaStreamDestroy(S->st); } if (S->ev) cudaEventDestroy(S->ev); if (S->h) cudaFreeHost(S->h); }
    for (int k = 0; k < G->nseg; k++) cudaFree(G->seg[k]);
    cudaFree(G->seg_d); cudaFree(G->ptab_d); cudaFree(G->X); cudaFree(G->Y); cudaFree(G->Z); cudaFree(G->N); cudaFree(G->need_d);
    cudaFree(G->cand); cudaFree(G->nc); cudaFree(G->ct0); cudaFree(G->prof); cudaFree(G->min); cudaFree(G->mout);
    if (G->st) cudaStreamDestroy(G->st);
    cudaGetLastError();
    free(G->ptab); free(G->dhave); free(G->seg); free(G->freel); free(G);
}
extern "C" int fgpu_broken(const fgpu *G) { return !G || G->broken; }
extern "C" void fgpu_set_broken(fgpu *G) { if (G) G->broken = 1; }

static int fg_slot(fgpu *G) {   /* under G->mu */
    if (!G->nfree) {
        if (G->nseg == G->cseg) return -1;
        uint8_t *p = nullptr;
        if (cudaMalloc(&p, SEG * CV) != cudaSuccess) { cudaGetLastError(); return -1; }
        cudaMemset(p, 0xa5, SEG * CV);   /* never-sent substreams read as garbage, not zeros (test_faces_gpu: a missed mark shows) */
        G->seg[G->nseg] = p;
        if (cudaMemcpy(G->seg_d + G->nseg, &p, sizeof p, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); cudaFree(p); return -1; }
        if (G->capfree < SEG * (G->nseg + 1)) { G->capfree = SEG * (G->nseg + 1); G->freel = (int *)realloc(G->freel, G->capfree * sizeof(int)); }
        for (int k = SEG - 1; k >= 0; k--) G->freel[G->nfree++] = G->nseg * SEG + k;
        G->nseg++;
    }
    return G->freel[--G->nfree];
}
/* the host chunk buffer (CS^3, z-major) of chunk id with its decoded substreams mask (bit s: z 16 (s / 4).., y 32 (s % 4)..; all ones:
   the whole chunk) -> the device; only substreams not uploaded yet are sent. Thread safe for distinct ids (the decode workers call it
   as each chunk is done): the new substreams are copied into one of NSTG pinned staging buffers and sent asynchronously from there. */
extern "C" int fgpu_put(fgpu *G, size_t id, const uint8_t *buf, uint32_t mask) {
    if (G->broken) return -1;
    pthread_mutex_lock(&G->mu);
    int s = G->ptab[id];
    if (s < 0) {
        if ((s = fg_slot(G)) < 0) { G->broken = 1; pthread_mutex_unlock(&G->mu); fprintf(stderr, "faces gpu: device chunk pool full (%d MB): CPU from here\n", G->nseg * (int)(SEG * CV >> 20)); return -1; }
        G->ptab[id] = s; G->dhave[id] = 0; G->dirty = 1;
    }
    const uint32_t add = mask & ~G->dhave[id];
    if (!add) { pthread_mutex_unlock(&G->mu); return 0; }
    G->dhave[id] |= add;
    int k = G->stg_next; G->stg_next = (k + 1) % NSTG;   /* a staging buffer: round robin, its own lock (wait for its last copy) */
    pthread_mutex_unlock(&G->mu);
    fg_stage *S = &G->stg[k];
    pthread_mutex_lock(&S->mu);
    cudaError_t e = cudaEventSynchronize(S->ev);
    uint8_t *d = G->seg[s / SEG] + (size_t)(s % SEG) * CV; size_t bytes = 0;
    if (e == cudaSuccess && add == 0xffffffffu) { memcpy(S->h, buf, CV); e = cudaMemcpyAsync(d, S->h, CV, cudaMemcpyHostToDevice, S->st); bytes = CV; }
    else for (int b = 0; b < 32 && e == cudaSuccess; b++) if (add >> b & 1) {
        const size_t o = ((size_t)16 * (b / 4) * CS + 32 * (b % 4)) * CS;
        for (int z = 0; z < 16; z++) memcpy(S->h + o + (size_t)z * CS * CS, buf + o + (size_t)z * CS * CS, 32 * CS);
        e = cudaMemcpy2DAsync(d + o, (size_t)CS * CS, S->h + o, (size_t)CS * CS, 32 * CS, 16, cudaMemcpyHostToDevice, S->st); bytes += 16 * 32 * CS;
    }
    if (e == cudaSuccess) e = cudaEventRecord(S->ev, S->st);
    pthread_mutex_unlock(&S->mu);
    if (e != cudaSuccess) { cudaGetLastError(); G->broken = 1; fprintf(stderr, "faces gpu: upload: %s: CPU from here\n", cudaGetErrorString(e)); return -1; }
    __atomic_fetch_add(&G->nput, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&G->bytes, bytes, __ATOMIC_RELAXED);
    return 0;
}
/* the device chunk of chunk id (a slot taken if it has none) for a decoder that writes it directly (gpudec_decode_dev); nullptr: pool
   full / broken. fgpu_have: substreams written there that way (fgpu_put then sends only the others) */
extern "C" uint8_t *fgpu_chunk_dev(fgpu *G, size_t id) {
    if (G->broken) return nullptr;
    pthread_mutex_lock(&G->mu);
    int s = G->ptab[id];
    if (s < 0) {
        if ((s = fg_slot(G)) < 0) { G->broken = 1; pthread_mutex_unlock(&G->mu); fprintf(stderr, "faces gpu: device chunk pool full (%d MB): CPU from here\n", G->nseg * (int)(SEG * CV >> 20)); return nullptr; }
        G->ptab[id] = s; G->dhave[id] = 0; G->dirty = 1;
    }
    uint8_t *d = G->seg[s / SEG] + (size_t)(s % SEG) * CV;
    pthread_mutex_unlock(&G->mu);
    return d;
}
extern "C" void fgpu_have(fgpu *G, size_t id, uint32_t mask) {
    pthread_mutex_lock(&G->mu);
    if (G->ptab[id] >= 0) G->dhave[id] |= mask;
    pthread_mutex_unlock(&G->mu);
}
static int fg_stage_wait(fgpu *G) {   /* every staged copy done (before a kernel reads the pool) */
    for (int k = 0; k < NSTG; k++) if (cudaStreamSynchronize(G->stg[k].st) != cudaSuccess) { cudaGetLastError(); G->broken = 1; return -1; }
    return 0;
}
extern "C" void fgpu_drop(fgpu *G, size_t id) {
    pthread_mutex_lock(&G->mu);
    const int s = G->ptab[id];
    if (s >= 0) { G->ptab[id] = -1; G->dhave[id] = 0; G->freel[G->nfree++] = s; G->dirty = 1; }
    pthread_mutex_unlock(&G->mu);
}

/* the wrap grid (w x h, Z < 0 invalid) and its outward normals N (3 per vertex, NaN: none) */
extern "C" int fgpu_grid(fgpu *G, const float *X, const float *Y, const float *Z, const float *N, int w, int h) {
    const size_t n = (size_t)w * h;
    if (n > G->ncap) {
        cudaFree(G->X); cudaFree(G->Y); cudaFree(G->Z); cudaFree(G->N); G->X = G->Y = G->Z = G->N = nullptr; G->ncap = 0;
        CK(cudaMalloc(&G->X, n * 4)); CK(cudaMalloc(&G->Y, n * 4)); CK(cudaMalloc(&G->Z, n * 4)); CK(cudaMalloc(&G->N, n * 12)); G->ncap = n;
    }
    G->n = n; G->w = w; G->h = h;
    CK(cudaMemcpyAsync(G->X, X, n * 4, cudaMemcpyHostToDevice, G->st)); CK(cudaMemcpyAsync(G->Y, Y, n * 4, cudaMemcpyHostToDevice, G->st));
    CK(cudaMemcpyAsync(G->Z, Z, n * 4, cudaMemcpyHostToDevice, G->st)); CK(cudaMemcpyAsync(G->N, N, n * 12, cudaMemcpyHostToDevice, G->st));
    return 0;
}

__device__ __forceinline__ int fg_vox(const int *__restrict__ ptab, uint8_t *const *__restrict__ seg, int g1, int g2, int64_t s0, int64_t s1, int64_t s2,
                                      int64_t z, int64_t y, int64_t x) {
    if (z < 0 || y < 0 || x < 0 || z >= s0 || y >= s1 || x >= s2) return -1;
    const int sl = ptab[((int)(z >> 7) * g1 + (int)(y >> 7)) * g2 + (int)(x >> 7)];
    if (sl < 0) return -1;
    return seg[sl / SEG][(size_t)(sl % SEG) * CV + (((int)(z & 127) * CS + (int)(y & 127)) * CS + (int)(x & 127))];
}

template <int MAXM>
__global__ void __launch_bounds__(128) k_band(const float *__restrict__ X, const float *__restrict__ Y, const float *__restrict__ Z, const float *__restrict__ N,
                                              int w, int r0, int nb, double f, double hs, double R, int m, double min_prom, double rel,
                                              const int *__restrict__ ptab, uint8_t *const *__restrict__ seg, int g1, int g2, int64_t s0, int64_t s1, int64_t s2,
                                              fc_band *cand, uint8_t *nc, float *ct0, uint8_t *prof, int ws) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x; if (li >= nb) return;
    const size_t i = (size_t)r0 * w + li; const int r = (int)(i / w), c = (int)(i % w);
    const float *Nv = &N[3 * i];
    if (!(Z[i] >= 0) || isnan(Nv[0])) { nc[li] = 0; ct0[li] = __int_as_float(0x7fc00000); return; }
    const double p0 = (Z[i] + 0.5) * f - 0.5, p1 = (Y[i] + 0.5) * f - 0.5, p2 = (X[i] + 0.5) * f - 0.5;
    const double n0 = Nv[0], n1 = Nv[1], n2 = Nv[2];
    float fv[MAXM]; int ex[MAXM];
    for (int k = 0; k < m; k++) {
        const double t = -R + k * hs, z = p0 + t * n0, y = p1 + t * n1, x = p2 + t * n2;
        const int64_t a = (int64_t)floor(z), b = (int64_t)floor(y), e = (int64_t)floor(x); const double fz = z - a, fy = y - b, fx = x - e;
        int v[8]; int bad = 0;
        for (int q = 0; q < 8; q++) { v[q] = fg_vox(ptab, seg, g1, g2, s0, s1, s2, a + (q >> 2 & 1), b + (q >> 1 & 1), e + (q & 1)); bad |= v[q] < 0; }
        fv[k] = bad ? __int_as_float(0x7fc00000) : (float)fc_trilin(v, fz, fy, fx);
    }
    ct0[li] = fv[(m - 1) / 2];
    if (r % SUB == 0 && c % SUB == 0) {
        uint8_t *q = &prof[((size_t)(r / SUB - r0 / SUB) * ws + c / SUB) * m];
        for (int k = 0; k < m; k++) q[k] = fv[k] == fv[k] ? (uint8_t)__float2int_rn(fminf(fmaxf(fv[k], 0), 255)) : 0;
    }
    nc[li] = (uint8_t)fc_bands_core(fv, ex, m, -R, hs, min_prom, rel, &cand[(size_t)li * FK], FK);
}

/* rows r0..r1 of the grid: profiles (m samples every hs CT voxels over -R..R along N; f: CT voxels per grid voxel) -> cand / nc / ct0
   (rows r0..r1 of the full arrays) and the kept profiles (prof: the full w/SUB x h/SUB x m array). 1: m too large for the GPU */
extern "C" int fgpu_band(fgpu *G, int r0, int r1, double f, double hs, double R, int m, double min_prom, double rel,
                         fc_band *cand, uint8_t *nc, float *ct0, uint8_t *prof) {
    if (G->broken) return -1;
    if (m > 1024) return 1;
    const int w = G->w, nb = (r1 - r0) * w, ws = (w + SUB - 1) / SUB, pr0 = r0 / SUB, pr1 = (r1 + SUB - 1) / SUB; const size_t pn = (size_t)(pr1 - pr0) * ws * m;
    if ((size_t)nb > G->bcap) {
        cudaFree(G->cand); cudaFree(G->nc); cudaFree(G->ct0); G->cand = nullptr; G->nc = nullptr; G->ct0 = nullptr; G->bcap = 0;
        CK(cudaMalloc(&G->cand, (size_t)nb * FK * sizeof(fc_band))); CK(cudaMalloc(&G->nc, nb)); CK(cudaMalloc(&G->ct0, (size_t)nb * 4)); G->bcap = nb;
    }
    if (pn > G->pcap) { cudaFree(G->prof); G->prof = nullptr; G->pcap = 0; CK(cudaMalloc(&G->prof, pn)); G->pcap = pn; }
    if (fg_stage_wait(G)) return -1;   /* the staged chunk uploads are in */
    if (G->dirty) { CK(cudaMemcpyAsync(G->ptab_d, G->ptab, G->ng * sizeof(int), cudaMemcpyHostToDevice, G->st)); G->dirty = 0; }
    CK(cudaMemsetAsync(G->prof, 0, pn, G->st));
    const int nbk = (nb + 127) / 128;
#define FG_L(M) k_band<M><<<nbk, 128, 0, G->st>>>(G->X, G->Y, G->Z, G->N, w, r0, nb, f, hs, R, m, min_prom, rel, G->ptab_d, G->seg_d, G->g[1], G->g[2], \
                                                    G->shape[0], G->shape[1], G->shape[2], G->cand, G->nc, G->ct0, G->prof, ws)
    if (m <= 128) FG_L(128); else if (m <= 256) FG_L(256); else if (m <= 512) FG_L(512); else FG_L(1024);
#undef FG_L
    CK(cudaGetLastError());
    /* rows with prof: copy the band's kept profiles into place (prof rows r0/SUB.. are this band's alone: r0 is a multiple of SUB) */
    CK(cudaMemcpyAsync(cand + (size_t)r0 * w * FK, G->cand, (size_t)nb * FK * sizeof(fc_band), cudaMemcpyDeviceToHost, G->st));
    CK(cudaMemcpyAsync(nc + (size_t)r0 * w, G->nc, nb, cudaMemcpyDeviceToHost, G->st));
    CK(cudaMemcpyAsync(ct0 + (size_t)r0 * w, G->ct0, (size_t)nb * 4, cudaMemcpyDeviceToHost, G->st));
    if (prof) CK(cudaMemcpyAsync(prof + (size_t)pr0 * ws * m, G->prof, pn, cudaMemcpyDeviceToHost, G->st));
    CK(cudaStreamSynchronize(G->st));
    return 0;
}

template <int CAP>
__global__ void __launch_bounds__(128) k_median(const float *__restrict__ in, float *out, int w, int h, int R, int minn) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t)w * h) return;
    const int r = (int)(i / w), c = (int)(i % w);
    float buf[CAP]; int k = 0;
    for (int a = r - R; a <= r + R; a++) { if (a < 0 || a >= h) continue; const float *row = in + (size_t)a * w;
        for (int b = c - R; b <= c + R; b++) if (b >= 0 && b < w && !isnan(row[b])) buf[k++] = row[b]; }
    out[i] = k >= minn ? fc_select(buf, k, k / 2) : __int_as_float(0x7fc00000);
}
/* the (2R + 1)^2 windowed median of in (NaN missing; NaN where fewer than minn values) -> out, as faces.c fc_median. 1: R too large */
extern "C" int fgpu_median(fgpu *G, const float *in, float *out, int w, int h, int R, int minn) {
    if (R > 20) return 1;
    const size_t n = (size_t)w * h;
    if (n > G->mcap) { cudaFree(G->min); cudaFree(G->mout); G->min = G->mout = nullptr; G->mcap = 0; CK(cudaMalloc(&G->min, n * 4)); CK(cudaMalloc(&G->mout, n * 4)); G->mcap = n; }
    CK(cudaMemcpyAsync(G->min, in, n * 4, cudaMemcpyHostToDevice, G->st));
    const unsigned nbk = (unsigned)((n + 127) / 128); const int ww = (2 * R + 1) * (2 * R + 1);
    if (ww <= 25) k_median<25><<<nbk, 128, 0, G->st>>>(G->min, G->mout, w, h, R, minn);
    else if (ww <= 81) k_median<81><<<nbk, 128, 0, G->st>>>(G->min, G->mout, w, h, R, minn);
    else if (ww <= 441) k_median<441><<<nbk, 128, 0, G->st>>>(G->min, G->mout, w, h, R, minn);
    else k_median<1681><<<nbk, 128, 0, G->st>>>(G->min, G->mout, w, h, R, minn);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(out, G->mout, n * 4, cudaMemcpyDeviceToHost, G->st));
    CK(cudaStreamSynchronize(G->st));
    return 0;
}
extern "C" void fgpu_stats(const fgpu *G, size_t *nput, size_t *bytes, int *slots) { *nput = G->nput; *bytes = G->bytes; *slots = G->nseg * SEG - G->nfree; }

/* the substreams the profiles of rows r0..r1 read: every in-volume trilinear corner of every sample (the positions of k_band) */
__global__ void __launch_bounds__(128) k_mark(const float *__restrict__ X, const float *__restrict__ Y, const float *__restrict__ Z, const float *__restrict__ N,
                                              int w, int r0, int nb, double f, double hs, double R, int m, int g1, int g2, int64_t s0, int64_t s1, int64_t s2, uint32_t *need) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x; if (li >= nb) return;
    const size_t i = (size_t)r0 * w + li; const float *Nv = &N[3 * i];
    if (!(Z[i] >= 0) || isnan(Nv[0])) return;
    const double p0 = (Z[i] + 0.5) * f - 0.5, p1 = (Y[i] + 0.5) * f - 0.5, p2 = (X[i] + 0.5) * f - 0.5;
    const double n0 = Nv[0], n1 = Nv[1], n2 = Nv[2];
    int last = -1; uint32_t lm = 0;
    for (int k = 0; k < m; k++) {
        const double t = -R + k * hs, z = p0 + t * n0, y = p1 + t * n1, x = p2 + t * n2;
        const int64_t a = (int64_t)floor(z), b = (int64_t)floor(y), e = (int64_t)floor(x);
        for (int q = 0; q < 8; q++) {
            const int64_t zz = a + (q >> 2 & 1), yy = b + (q >> 1 & 1), xx = e + (q & 1);
            if (zz < 0 || yy < 0 || xx < 0 || zz >= s0 || yy >= s1 || xx >= s2) continue;
            const int id = ((int)(zz >> 7) * g1 + (int)(yy >> 7)) * g2 + (int)(xx >> 7);
            const uint32_t bit = 1u << (((int)(zz & 127) >> 4) * 4 + ((int)(yy & 127) >> 5));
            if (id != last) { if (last >= 0) atomicOr(&need[last], lm); last = id; lm = 0; }
            lm |= bit;
        }
    }
    if (last >= 0) atomicOr(&need[last], lm);
}
/* rows r0..r1: per chunk the substream bits the profiles will read -> mask[ng] (host; zeroed device side afterwards) */
extern "C" int fgpu_mark(fgpu *G, int r0, int r1, double f, double hs, double R, int m, uint32_t *mask) {
    if (G->broken) return -1;
    const int nb = (r1 - r0) * G->w;
    k_mark<<<(nb + 127) / 128, 128, 0, G->st>>>(G->X, G->Y, G->Z, G->N, G->w, r0, nb, f, hs, R, m, G->g[1], G->g[2], G->shape[0], G->shape[1], G->shape[2], G->need_d);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(mask, G->need_d, G->ng * sizeof(uint32_t), cudaMemcpyDeviceToHost, G->st));
    CK(cudaMemsetAsync(G->need_d, 0, G->ng * sizeof(uint32_t), G->st));
    CK(cudaStreamSynchronize(G->st));
    return 0;
}
extern "C" size_t fgpu_nchunks(const fgpu *G) { return G->ng; }
