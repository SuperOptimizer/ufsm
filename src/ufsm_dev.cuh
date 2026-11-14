#pragma once
/* ufsm_dev.cuh (2026-11-14, ops audit #8): logical devices. Force-included into every CUDA unit (CMakeLists.txt: -include /
   --pre-include), so all device selection goes through one map: logical device d runs on physical device ufsm_dev_phys(d).
   The map is the identity unless nn_set_devmap() set one (train --gpus 0,0: two logical devices on one card), so ordinary runs see
   no change. With a map, every per-device table in the GPU code (streams, plans, buffers, kernel attributes: indexed by
   cudaGetDevice() / cur_dev()) is indexed by the LOGICAL device, so two data-parallel ranks on one card keep separate state as on
   two cards. That lets the 1-GPU perf pod run the 2-GPU data-parallel tests (test_univ2, the DP equivalence test), which no machine
   could run while the desk's cards belonged to scratch2.
   Wrapped: cudaSetDevice / cudaGetDevice (a thread's logical device), cudaGetDeviceCount (at least the logical count),
   cudaDeviceGetAttribute / cudaGetDeviceProperties / cudaMemGetInfo (free memory split among the logical devices sharing a card),
   cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess / cudaMemcpyPeer(Async) (one card: a plain device copy). */
#include <cuda_runtime.h>

/* the map lives here as C++17 inline variables (one definition across all units, whatever objects a test links: a definition in
   nn_core.cu left test binaries that link only some CUDA objects without it); nn_set_devmap (nn_core.cu) fills it. mgpu: process */
inline int ufsm_devmap_[16];
inline int ufsm_ndevmap_ = 0;
inline thread_local int ufsm_t_ldev = -1;   /* this thread's logical device (-1: never set: the physical one is its logical one) */
static inline int ufsm_dev_phys(int d) { return ufsm_ndevmap_ > 0 && d >= 0 && d < ufsm_ndevmap_ ? ufsm_devmap_[d] : d; }
static inline int ufsm_dev_nlogical(void) { return ufsm_ndevmap_; }
static inline int ufsm_dev_share(int d) { if (ufsm_ndevmap_ <= 0) return 1; int n = 0; for (int k = 0; k < ufsm_ndevmap_; k++) n += ufsm_devmap_[k] == ufsm_dev_phys(d); return n > 0 ? n : 1; }

static inline cudaError_t ufsm_cudaSetDevice(int d) { const cudaError_t e = cudaSetDevice(ufsm_dev_phys(d)); if (e == cudaSuccess) ufsm_t_ldev = d; return e; }
static inline cudaError_t ufsm_cudaGetDevice(int *d) {
    if (ufsm_t_ldev >= 0 && ufsm_dev_nlogical() > 0) { *d = ufsm_t_ldev; return cudaSuccess; }
    return cudaGetDevice(d);
}
static inline cudaError_t ufsm_cudaGetDeviceCount(int *n) { const cudaError_t e = cudaGetDeviceCount(n); if (e == cudaSuccess && ufsm_dev_nlogical() > *n) *n = ufsm_dev_nlogical(); return e; }
static inline cudaError_t ufsm_cudaDeviceGetAttribute(int *v, cudaDeviceAttr a, int d) { return cudaDeviceGetAttribute(v, a, ufsm_dev_phys(d)); }
static inline cudaError_t ufsm_cudaGetDeviceProperties(cudaDeviceProp *p, int d) { return cudaGetDeviceProperties(p, ufsm_dev_phys(d)); }
static inline cudaError_t ufsm_cudaMemGetInfo(size_t *fr, size_t *tot) {
    const cudaError_t e = cudaMemGetInfo(fr, tot); int d = 0; ufsm_cudaGetDevice(&d); const int s = ufsm_dev_share(d);
    if (e == cudaSuccess && s > 1) { *fr /= s; *tot /= s; }
    return e;
}
static inline cudaError_t ufsm_cudaDeviceCanAccessPeer(int *c, int d, int p) {
    if (ufsm_dev_phys(d) == ufsm_dev_phys(p)) { *c = 1; return cudaSuccess; }
    return cudaDeviceCanAccessPeer(c, ufsm_dev_phys(d), ufsm_dev_phys(p));
}
static inline cudaError_t ufsm_cudaDeviceEnablePeerAccess(int p, unsigned f) {
    int d = 0; ufsm_cudaGetDevice(&d); if (ufsm_dev_phys(d) == ufsm_dev_phys(p)) return cudaSuccess;
    return cudaDeviceEnablePeerAccess(ufsm_dev_phys(p), f);
}
static inline cudaError_t ufsm_cudaMemcpyPeer(void *dst, int dd, const void *src, int sd, size_t n) {
    if (ufsm_dev_phys(dd) == ufsm_dev_phys(sd)) return cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice);
    return cudaMemcpyPeer(dst, ufsm_dev_phys(dd), src, ufsm_dev_phys(sd), n);
}
static inline cudaError_t ufsm_cudaMemcpyPeerAsync(void *dst, int dd, const void *src, int sd, size_t n, cudaStream_t s) {
    if (ufsm_dev_phys(dd) == ufsm_dev_phys(sd)) return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s);
    return cudaMemcpyPeerAsync(dst, ufsm_dev_phys(dd), src, ufsm_dev_phys(sd), n, s);
}
#undef cudaSetDevice
#define cudaSetDevice ufsm_cudaSetDevice
#undef cudaGetDevice
#define cudaGetDevice ufsm_cudaGetDevice
#undef cudaGetDeviceCount
#define cudaGetDeviceCount ufsm_cudaGetDeviceCount
#undef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute ufsm_cudaDeviceGetAttribute
#undef cudaGetDeviceProperties
#define cudaGetDeviceProperties ufsm_cudaGetDeviceProperties
#undef cudaMemGetInfo
#define cudaMemGetInfo ufsm_cudaMemGetInfo
#undef cudaDeviceCanAccessPeer
#define cudaDeviceCanAccessPeer ufsm_cudaDeviceCanAccessPeer
#undef cudaDeviceEnablePeerAccess
#define cudaDeviceEnablePeerAccess ufsm_cudaDeviceEnablePeerAccess
#undef cudaMemcpyPeer
#define cudaMemcpyPeer ufsm_cudaMemcpyPeer
#undef cudaMemcpyPeerAsync
#define cudaMemcpyPeerAsync ufsm_cudaMemcpyPeerAsync
