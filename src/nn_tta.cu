/* predict --tta: the window symmetries of src/tta.h on the device (CT in, logits back in place) */
#include "nn_common.cuh"
#include "tta.h"

__device__ __forceinline__ size_t tta_idx(int code, int D, int W, size_t v) {
    int z = (int)(v / ((size_t)W * W)), y = (int)(v / W % W), x = (int)(v % W);
    if (code & TTA_FZ) z = D - 1 - z;
    if (code & TTA_FY) y = W - 1 - y;
    if (code & TTA_FX) x = W - 1 - x;
    if (code & TTA_SWAP) { const int t = y; y = x; x = t; }
    return ((size_t)z * W + y) * W + x;
}
__global__ void tta_ct_k(const uint8_t *src, uint8_t *dst, int D, int W, int code) {
    const size_t n = (size_t)D * W * W, v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v < n) dst[v] = src[tta_idx(code, D, W, v)];
}
struct tta_map_t { int nch, src[32], neg[32]; };
/* in place: each thread owns its voxel v of every channel c and the partner element (src[c], T v): every element is read and
   written by exactly one thread (T and the channel map are involutions); a channel mapped to itself swaps the pair once (v < T v) */
template <typename U> __global__ void tta_lg_k(U *lg, int D, int W, int code, tta_map_t m, U signbit) {
    const size_t n = (size_t)D * W * W, v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (v >= n) return;
    const size_t u = tta_idx(code, D, W, v);
    for (int c = 0; c < m.nch; c++) {
        const int c2 = m.src[c];
        if (c2 < c) continue;
        U *a = lg + (size_t)c * n, *b = lg + (size_t)c2 * n;
        const U sa = m.neg[c] ? signbit : (U)0, sb = m.neg[c2] ? signbit : (U)0;
        if (c2 == c) {
            if (u < v) continue;
            if (u == v) { a[v] ^= sa; continue; }
            const U t = a[v]; a[v] = a[u] ^ sa; a[u] = t ^ sa;
        } else { const U t = a[v]; a[v] = b[u] ^ sa; b[u] = t ^ sb; }
    }
}
extern "C" void nn_tta_ct(const uint8_t *src, uint8_t *dst, int D, int W, int code) {
    const size_t n = (size_t)D * W * W;
    tta_ct_k<<<nblk(n, 256), 256>>>(src, dst, D, W, code); KCHECK();
}
extern "C" void nn_tta_logits(void *lg, int h16, int D, int W, int code, int nch, const int *src, const int *sgn) {
    if (!code) return;
    tta_map_t m; m.nch = nch < 32 ? nch : 32;
    for (int c = 0; c < m.nch; c++) { m.src[c] = src[c]; m.neg[c] = sgn[c] < 0; }
    const size_t n = (size_t)D * W * W;
    if (h16) tta_lg_k<uint16_t><<<nblk(n, 256), 256>>>((uint16_t *)lg, D, W, code, m, (uint16_t)0x8000u);
    else tta_lg_k<uint32_t><<<nblk(n, 256), 256>>>((uint32_t *)lg, D, W, code, m, 0x80000000u);
    KCHECK();
}
