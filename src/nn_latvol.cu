/* latvol on the GPU (latvol.h, nn_latvol.h): the chunk streams of a window decoded straight into the device cell tensor the trunk's
 * input kernel reads, the cells' CT bytes, and the forward transform of an encoder. Byte-identical to src/latvol.c: the entropy decode
 * is the same integer code (latvol.h's lv_dec_*), the transforms are int32 sums with one float multiply and a rint (built without fast
 * math and with FMA contraction off, CMakeLists.txt UFSM_STRICT_UNITS).
 *   k_offs:   one thread per chunk: the header check and the 512 substreams' word offsets (a prefix sum of the header's lengths).
 *   k_decode: one thread per (chunk, substream) whose 4^3 cells meet the window: the serial tANS / raw-bit decode of its 64 cells
 *             (<= C symbols each), every cell written to the window's int8 cells as it is decoded. Substreams outside the window, air
 *             substreams and absent chunks cost nothing (the tensor is zeroed first: zero cells are air).
 *   k_synth:  one thread per cell: its 64 CT bytes (0 in air cells).
 *   k_fwd:    one thread per cell: its C indices from the 64 CT bytes. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "latvol.h"
#include "nn_latvol.h"

#define LVCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "latvol gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

struct nn_lv {
    int C, ntab; float step_raw;
    lv_dent *dt; uint8_t *map; int16_t *bi;      /* device copies */
    uint32_t *offs; lv_chunk_ref *refs; int cap; /* per-chunk scratch: [cap][LV_NSUB] word offsets, [cap] refs */
    int *flag;
    double t_dec; long n_dec;                    /* nn_latvol_stats */
};
static double lv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
extern "C" void nn_latvol_stats(const nn_lv *L, double *seconds, long *calls) { *seconds = L->t_dec; *calls = L->n_dec; }

extern "C" nn_lv *nn_latvol_create(const uint32_t *dt, int ntab, const uint8_t *map, const int16_t *bi, int C, float step_raw) {
    nn_lv *L = (nn_lv *)calloc(1, sizeof *L); if (!L) return nullptr;
    L->C = C; L->ntab = ntab; L->step_raw = step_raw;
    LVCK(cudaMalloc(&L->dt, (size_t)ntab * LV_L * sizeof(lv_dent))); LVCK(cudaMemcpy(L->dt, dt, (size_t)ntab * LV_L * sizeof(lv_dent), cudaMemcpyHostToDevice));
    LVCK(cudaMalloc(&L->map, LV_MAXC)); LVCK(cudaMemset(L->map, 0, LV_MAXC)); LVCK(cudaMemcpy(L->map, map, (size_t)C, cudaMemcpyHostToDevice));
    LVCK(cudaMalloc(&L->bi, (size_t)C * 64 * sizeof(int16_t))); LVCK(cudaMemcpy(L->bi, bi, (size_t)C * 64 * sizeof(int16_t), cudaMemcpyHostToDevice));
    LVCK(cudaMalloc(&L->flag, sizeof(int)));
    return L;
}
extern "C" void nn_latvol_free(nn_lv *L) {
    if (!L) return;
    cudaFree(L->dt); cudaFree(L->map); cudaFree(L->bi); cudaFree(L->offs); cudaFree(L->refs); cudaFree(L->flag); free(L);
}

__global__ void k_offs(const uint8_t *comp, const lv_chunk_ref *refs, int nchunk, int C, uint32_t *offs, int *flag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nchunk) return;
    const uint8_t *h = comp + refs[i].off; const uint32_t *hw = (const uint32_t *)h; const uint16_t *words = (const uint16_t *)(h + 8);
    if (hw[0] != LV_MAGIC || h[4] != 1 || h[5] != C) { *flag = 1; for (int s = 0; s < LV_NSUB; s++) offs[(size_t)i * LV_NSUB + s] = 0xffffffffu; return; }
    uint32_t o = 0;
    for (int s = 0; s < LV_NSUB; s++) { const uint32_t nw = words[s]; offs[(size_t)i * LV_NSUB + s] = nw <= LV_SUB_WORDS && (size_t)LV_HDR + ((size_t)o + nw) * 4 <= refs[i].len ? o : 0xffffffffu; o += nw; }
    if ((size_t)LV_HDR + (size_t)o * 4 != refs[i].len) *flag = 1;
}
__global__ void k_decode(const uint8_t *comp, const lv_chunk_ref *refs, int nchunk, const uint32_t *offs, const lv_dent *dt, const uint8_t *mapd, int C,
                         int d, int h, int w, int stride, int8_t *out, int *flag) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nchunk * LV_NSUB) return;
    const int ci = (int)(i / LV_NSUB), s = (int)(i % LV_NSUB);
    const lv_chunk_ref rf = refs[ci];
    const int z0 = rf.cz + 4 * (s >> 6), y0 = rf.cy + 4 * (s >> 3 & 7), x0 = rf.cx + 4 * (s & 7);
    if (z0 + 4 <= 0 || y0 + 4 <= 0 || x0 + 4 <= 0 || z0 >= d || y0 >= h || x0 >= w) return;
    const uint8_t *hd = comp + rf.off; const uint32_t nw = ((const uint16_t *)(hd + 8))[s];
    if (!nw) return;
    const uint32_t o = offs[i];
    if (o == 0xffffffffu) { *flag = 1; return; }
    uint8_t map[LV_MAXC];
    for (int c = 0; c < C; c++) map[c] = mapd[c];
    lv_dec dc;
    if (lv_dec_init(&dc, (const uint32_t *)(hd + LV_HDR) + o, nw)) { *flag = 1; return; }
    const int na = C < stride - 1 ? C : stride - 1;
    for (int cell = 0; cell < LV_SUBCELLS; cell++) {
        int16_t qc[LV_MAXC];
        if (!lv_dec_cell(&dc, dt, map, C, qc)) continue;
        const int z = z0 + (cell >> 4), y = y0 + (cell >> 2 & 3), x = x0 + (cell & 3);
        if (z < 0 || y < 0 || x < 0 || z >= d || y >= h || x >= w) continue;
        int8_t *p = out + (((size_t)z * h + y) * w + x) * stride;
        p[0] = (int8_t)(uint8_t)(qc[0] & 255); p[1] = (int8_t)(qc[0] >> 8);
        for (int c = 1; c < na; c++) p[c + 1] = (int8_t)qc[c];
    }
    if (lv_dec_end(&dc)) *flag = 1;
}
extern "C" int nn_latvol_decode(nn_lv *L, const void *comp_d, const lv_chunk_ref *refs_h, int nchunk, int d, int h, int w, int stride, void *cells_d) {
    const double t0 = lv_now();
    LVCK(cudaMemsetAsync(cells_d, 0, (size_t)d * h * w * stride));
    if (nchunk <= 0) return 0;
    if (nchunk > L->cap) {
        cudaFree(L->offs); cudaFree(L->refs); L->cap = nchunk + nchunk / 2 + 8;
        LVCK(cudaMalloc(&L->offs, (size_t)L->cap * LV_NSUB * sizeof(uint32_t))); LVCK(cudaMalloc(&L->refs, (size_t)L->cap * sizeof(lv_chunk_ref)));
    }
    LVCK(cudaMemcpy(L->refs, refs_h, (size_t)nchunk * sizeof(lv_chunk_ref), cudaMemcpyHostToDevice));
    LVCK(cudaMemsetAsync(L->flag, 0, sizeof(int)));
    k_offs<<<(nchunk + 63) / 64, 64>>>((const uint8_t *)comp_d, L->refs, nchunk, L->C, L->offs, L->flag);
    k_decode<<<(unsigned)(((size_t)nchunk * LV_NSUB + 127) / 128), 128>>>((const uint8_t *)comp_d, L->refs, nchunk, L->offs, L->dt, L->map, L->C, d, h, w, stride, (int8_t *)cells_d, L->flag);
    LVCK(cudaGetLastError());
    int f = 0; LVCK(cudaMemcpy(&f, L->flag, sizeof f, cudaMemcpyDeviceToHost));
    L->t_dec += lv_now() - t0; L->n_dec++;
    return f;
}

__global__ void k_synth(const int8_t *q, int stride, int C, const int16_t *bi, float rs, int d, int h, int w, uint8_t *ct) {
    const size_t cv = (size_t)d * h * w, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * w)), y = (int)(i / w % h), x = (int)(i % w), H = 4 * h, W = 4 * w;
    const int8_t *qc = q + i * stride;
    const int q0 = (int)(uint8_t)qc[0] + 256 * (int)qc[1];
    int32_t acc[64];
    for (int k = 0; k < 64; k++) acc[k] = q0 * (int32_t)bi[k];
    for (int c = 1; c < C; c++) { const int32_t v = qc[c + 1]; if (v) for (int k = 0; k < 64; k++) acc[k] += v * (int32_t)bi[c * 64 + k]; }
    for (int k = 0; k < 64; k++) ct[((size_t)(4 * z + (k >> 4)) * H + 4 * y + (k >> 2 & 3)) * W + 4 * x + (k & 3)] = q0 ? lv_inv_byte(acc[k], rs) : (uint8_t)0;
}
extern "C" void nn_latvol_synth(nn_lv *L, const void *cells_d, int stride, int d, int h, int w, uint8_t *ct_d) {
    if (stride < L->C + 1) { fprintf(stderr, "nn_latvol_synth: the cells hold %d of the store's %d channels\n", stride - 1, L->C); abort(); }
    const size_t cv = (size_t)d * h * w;
    k_synth<<<(unsigned)((cv + 127) / 128), 128>>>((const int8_t *)cells_d, stride, L->C, L->bi, L->step_raw / (float)(1 << LV_BSHIFT), d, h, w, ct_d);
    LVCK(cudaGetLastError());
}

__global__ void k_fwd(const uint8_t *ct, int C, const int16_t *bi, float fs, int d, int h, int w, int16_t *q) {
    const size_t cv = (size_t)d * h * w, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cv) return;
    const int z = (int)(i / ((size_t)h * w)), y = (int)(i / w % h), x = (int)(i % w), H = 4 * h, W = 4 * w;
    int32_t v[64]; int any = 0;
    for (int k = 0; k < 64; k++) { v[k] = ct[((size_t)(4 * z + (k >> 4)) * H + 4 * y + (k >> 2 & 3)) * W + 4 * x + (k & 3)]; any |= v[k]; }
    int16_t *o = q + i * C;
    if (!any) { for (int c = 0; c < C; c++) o[c] = 0; return; }
    int nz = 0;
    for (int c = 0; c < C; c++) { int32_t acc = 0; for (int k = 0; k < 64; k++) acc += (int32_t)bi[c * 64 + k] * v[k]; o[c] = (int16_t)lv_fwd_index(acc, fs, c); nz |= c && o[c]; }
    o[0] = (int16_t)lv_fwd_mean(o[0], nz);
}
extern "C" void nn_latvol_forward(nn_lv *L, const uint8_t *ct_d, int d, int h, int w, int16_t *q_d) {
    const size_t cv = (size_t)d * h * w;
    k_fwd<<<(unsigned)((cv + 127) / 128), 128>>>(ct_d, L->C, L->bi, 1.0f / ((float)(1 << LV_BSHIFT) * L->step_raw), d, h, w, q_d);
    LVCK(cudaGetLastError());
}

/* ---- a box of a latvol array as CT bytes, from any thread (the voxel reader below z3_read: src/latvol_read.c lv_box_voxels_gpu) ----
   A pool of `slots` contexts on one device, each with its own stream and buffers; a caller takes a free one (waits when all are busy).
   Each context keeps the device tables of the codecs it has seen (keyed by the host codec's address). The bytes are nn_latvol_decode +
   nn_latvol_synth's (the same kernels on the context's stream). */
enum { LB_COMP, LB_CELLS, LB_CT, LB_OFFS, LB_REFS, LB_N, LB_TAB = 8 };
typedef struct { const void *key; lv_dent *dt; uint8_t *map; int16_t *bi; } lb_tab;
typedef struct { int busy, init; cudaStream_t st; struct { void *p; size_t cap; } b[LB_N]; int *flag; lb_tab tab[LB_TAB]; int ntab; } lb_ctx;
static struct { pthread_mutex_t mu; pthread_cond_t cv; int dev, slots; lb_ctx c[16]; } g_lb = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, -1, 2, {}};
extern "C" void nn_latvol_box_config(int dev, int slots) { pthread_mutex_lock(&g_lb.mu); g_lb.dev = dev; g_lb.slots = slots < 1 ? 1 : slots > 16 ? 16 : slots; pthread_mutex_unlock(&g_lb.mu); }
#define LBCK(x) do { if ((x) != cudaSuccess) { rc = -1; goto out; } } while (0)
static int lb_grow(lb_ctx *x, int i, size_t n) {
    if (x->b[i].cap >= n) return 0;
    if (x->b[i].p) cudaFree(x->b[i].p);
    x->b[i].p = nullptr; x->b[i].cap = 0; const size_t c = n + n / 8 + 4096;
    if (cudaMalloc(&x->b[i].p, c) != cudaSuccess) return -1;
    x->b[i].cap = c; return 0;
}
extern "C" int nn_latvol_box(const void *key, const uint32_t *dt, int ntab, const uint8_t *map, const int16_t *bi, int C, float step_raw,
                             const uint8_t *comp, size_t ncomp, const lv_chunk_ref *refs, int nref, int d, int h, int w, uint8_t *ct_out) {
    pthread_mutex_lock(&g_lb.mu);
    if (g_lb.dev < 0) { pthread_mutex_unlock(&g_lb.mu); return -1; }
    lb_ctx *x = nullptr;
    for (;;) { for (int i = 0; i < g_lb.slots && !x; i++) if (!g_lb.c[i].busy) x = &g_lb.c[i]; if (x) break; pthread_cond_wait(&g_lb.cv, &g_lb.mu); }
    x->busy = 1; const int dev = g_lb.dev;
    pthread_mutex_unlock(&g_lb.mu);
    int rc = 0, f = 0; lb_tab *t = nullptr; const int stride = C + 1; const size_t cv = (size_t)d * h * w;
    LBCK(cudaSetDevice(dev));
    if (!x->init) { LBCK(cudaStreamCreateWithFlags(&x->st, cudaStreamNonBlocking)); LBCK(cudaMalloc(&x->flag, sizeof(int))); x->init = 1; }
    for (int i = 0; i < x->ntab; i++) if (x->tab[i].key == key) t = &x->tab[i];
    if (!t) {
        if (x->ntab == LB_TAB) { for (int i = 0; i < LB_TAB; i++) { cudaFree(x->tab[i].dt); cudaFree(x->tab[i].map); cudaFree(x->tab[i].bi); } x->ntab = 0; }
        t = &x->tab[x->ntab]; t->dt = nullptr; t->map = nullptr; t->bi = nullptr;
        LBCK(cudaMalloc(&t->dt, (size_t)ntab * LV_L * sizeof(lv_dent))); LBCK(cudaMalloc(&t->map, LV_MAXC)); LBCK(cudaMalloc(&t->bi, (size_t)C * 64 * sizeof(int16_t)));
        LBCK(cudaMemcpy(t->dt, dt, (size_t)ntab * LV_L * sizeof(lv_dent), cudaMemcpyHostToDevice)); LBCK(cudaMemset(t->map, 0, LV_MAXC)); LBCK(cudaMemcpy(t->map, map, (size_t)C, cudaMemcpyHostToDevice));
        LBCK(cudaMemcpy(t->bi, bi, (size_t)C * 64 * sizeof(int16_t), cudaMemcpyHostToDevice));
        t->key = key; x->ntab++;
    }
    if (lb_grow(x, LB_COMP, ncomp) || lb_grow(x, LB_CELLS, cv * stride) || lb_grow(x, LB_CT, cv * 64) || lb_grow(x, LB_OFFS, (size_t)nref * LV_NSUB * sizeof(uint32_t)) || lb_grow(x, LB_REFS, (size_t)nref * sizeof(lv_chunk_ref))) { rc = -1; goto out; }
    LBCK(cudaMemcpyAsync(x->b[LB_COMP].p, comp, ncomp, cudaMemcpyHostToDevice, x->st));
    LBCK(cudaMemcpyAsync(x->b[LB_REFS].p, refs, (size_t)nref * sizeof(lv_chunk_ref), cudaMemcpyHostToDevice, x->st));
    LBCK(cudaMemsetAsync(x->b[LB_CELLS].p, 0, cv * stride, x->st)); LBCK(cudaMemsetAsync(x->flag, 0, sizeof(int), x->st));
    k_offs<<<(nref + 63) / 64, 64, 0, x->st>>>((const uint8_t *)x->b[LB_COMP].p, (const lv_chunk_ref *)x->b[LB_REFS].p, nref, C, (uint32_t *)x->b[LB_OFFS].p, x->flag);
    k_decode<<<(unsigned)(((size_t)nref * LV_NSUB + 127) / 128), 128, 0, x->st>>>((const uint8_t *)x->b[LB_COMP].p, (const lv_chunk_ref *)x->b[LB_REFS].p, nref, (const uint32_t *)x->b[LB_OFFS].p, t->dt, t->map, C, d, h, w, stride, (int8_t *)x->b[LB_CELLS].p, x->flag);
    k_synth<<<(unsigned)((cv + 127) / 128), 128, 0, x->st>>>((const int8_t *)x->b[LB_CELLS].p, stride, C, t->bi, step_raw / (float)(1 << LV_BSHIFT), d, h, w, (uint8_t *)x->b[LB_CT].p);
    LBCK(cudaGetLastError());
    LBCK(cudaMemcpyAsync(ct_out, x->b[LB_CT].p, cv * 64, cudaMemcpyDeviceToHost, x->st));
    LBCK(cudaMemcpyAsync(&f, x->flag, sizeof f, cudaMemcpyDeviceToHost, x->st));
    LBCK(cudaStreamSynchronize(x->st));
    if (f) rc = 1;
out:
    pthread_mutex_lock(&g_lb.mu); x->busy = 0; pthread_cond_signal(&g_lb.cv); pthread_mutex_unlock(&g_lb.mu);
    return rc;
}
