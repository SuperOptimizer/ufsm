/* predict's device-resident CT window (UFSM_PRED_DEVREAD): reader workers decode whole aligned 128^3 chunks with z3_read_sub_dev (the
   GPU's substreams straight into a device chunk, the rest (CPU fallbacks: a busy GPU, corrupt or non-lossy chunks, fill) in a host chunk
   with zeros at the device ones), and each chunk's part of the window is gathered on the device, substream by substream from the device
   or the uploaded host chunk. Each worker has its own non-blocking stream (the legacy default stream would serialise with predict's
   compute stream) and returns when its chunk is in the window. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { cudaStream_t st; uint8_t *dch, *dh, *hch; int dev; } ctw_t;
extern "C" void *nn_ctw_new(int dev) {
    ctw_t *w = (ctw_t *)calloc(1, sizeof *w); if (!w) return nullptr;
    w->dev = dev; cudaSetDevice(dev);
    if (cudaStreamCreateWithFlags(&w->st, cudaStreamNonBlocking) != cudaSuccess || cudaMalloc(&w->dch, 1u << 21) != cudaSuccess ||
        cudaMalloc(&w->dh, 1u << 21) != cudaSuccess || cudaMallocHost(&w->hch, 1u << 21) != cudaSuccess) { cudaGetLastError(); free(w); return nullptr; }
    return w;
}
extern "C" void nn_ctw_free(void *h) {
    ctw_t *w = (ctw_t *)h; if (!w) return;
    cudaSetDevice(w->dev); cudaStreamSynchronize(w->st); cudaFree(w->dch); cudaFree(w->dh); cudaFreeHost(w->hch); cudaStreamDestroy(w->st); free(w);
}
extern "C" uint8_t *nn_ctw_host(void *h) { return ((ctw_t *)h)->hch; }
extern "C" uint8_t *nn_ctw_dev(void *h) { return ((ctw_t *)h)->dch; }
/* window voxel (z, y, x) of the box [a, b) (window coordinates) from chunk-local (z + d0, y + d1, x + d2): substream (cz / 16) * 4 + cy / 32 */
__global__ static void ctw_gather_k(const uint8_t *__restrict__ dch, const uint8_t *__restrict__ dh, uint32_t dm, int a0, int a1, int a2, int n0, int n1, int n2,
                                    int d0, int d1, int d2, int wy, int wx, uint8_t *__restrict__ win) {
    const size_t N = (size_t)n0 * n1 * n2;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(i % n2), y = (int)(i / n2 % n1), z = (int)(i / ((size_t)n1 * n2));
        const int cz = z + d0, cy = y + d1, cx = x + d2; const size_t ci = ((size_t)cz * 128 + cy) * 128 + cx;
        const uint32_t s = (uint32_t)(cz / 16 * 4 + cy / 32);
        win[((size_t)(z + a0) * wy + (y + a1)) * wx + (x + a2)] = dm >> s & 1u ? dch[ci] : dh ? dh[ci] : 0;
    }
}
/* the chunk at chunk-origin co (array coordinates; its device part dm, its host part in nn_ctw_host when host != 0) into the window at
   origin wo of extent wn, clipped to the window and to the array shape sh; waits for the copy. 0 ok */
extern "C" int nn_ctw_put(void *h, uint32_t dm, int host, const int64_t co[3], const int64_t wo[3], const int64_t wn[3], const int64_t sh[3], uint8_t *dwin) {
    ctw_t *w = (ctw_t *)h; int a[3], n[3], d[3];
    for (int k = 0; k < 3; k++) {
        int64_t lo = co[k] > wo[k] ? co[k] : wo[k], hi = co[k] + 128 < wo[k] + wn[k] ? co[k] + 128 : wo[k] + wn[k]; if (hi > sh[k]) hi = sh[k];
        if (hi <= lo) return 0;
        a[k] = (int)(lo - wo[k]); n[k] = (int)(hi - lo); d[k] = (int)(lo - co[k]);
    }
    cudaSetDevice(w->dev);
    if (host && cudaMemcpyAsync(w->dh, w->hch, 1u << 21, cudaMemcpyHostToDevice, w->st) != cudaSuccess) return -1;
    const size_t N = (size_t)n[0] * n[1] * n[2]; const unsigned nb = (unsigned)((N + 255) / 256 < 4096 ? (N + 255) / 256 : 4096);
    ctw_gather_k<<<nb, 256, 0, w->st>>>(w->dch, host ? w->dh : nullptr, dm, a[0], a[1], a[2], n[0], n[1], n[2], d[0], d[1], d[2], (int)wn[1], (int)wn[2], dwin);
    return cudaStreamSynchronize(w->st) == cudaSuccess ? 0 : -1;
}
extern "C" int nn_ctw_zero(void *h, uint8_t *d, size_t n) {   /* (on the worker's stream, waited for) */
    ctw_t *w = (ctw_t *)h; cudaSetDevice(w->dev);
    return cudaMemsetAsync(d, 0, n, w->st) == cudaSuccess && cudaStreamSynchronize(w->st) == cudaSuccess ? 0 : -1;
}
extern "C" int nn_dev_mem(size_t *fr, size_t *tot) {   /* the current device's free / total bytes */
    if (cudaMemGetInfo(fr, tot) != cudaSuccess) { cudaGetLastError(); *fr = *tot = 0; return -1; }
    return 0;
}
