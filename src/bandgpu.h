/* band_field_phase_pair on the GPU (src/nn_band.cu): the two parity EDTs (feature transform, the same Felzenszwalb-Huttenlocher
   envelopes as src/band.c edt_feature_cap, lean: indices only, squared distances rebuilt from them) and the pair rule's phase per
   voxel. Built without fast math and with -fmad=false (Makefile), so every float operation rounds as band.c's (gcc -std=c23 does
   not contract either): the output is byte-identical to band_field_phase_pair (tests/test_band_pair.c). Declared weak: a binary
   without nn_band.o has a null band_pair_gpu and runs the CPU code. Device buffers and a stream per slot
   (a pool of --gpu-slots contexts, grown on demand; callers wait for a free one), so N slots hold ~10 bytes a block voxel
   each on the device. Returns 0 on success, nonzero on any CUDA failure (the caller falls back to the CPU). */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((weak)) int band_pair_gpu(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase);
__attribute__((weak)) void band_pair_gpu_slots(int slots, int device);   /* before the first call: concurrent callers, the device */
__attribute__((weak)) void band_pair_gpu_free(void);                     /* the idle slots' buffers */
__attribute__((weak)) int band_pair_gpu_reserve(int edge);
__attribute__((weak)) void *band_pair_gpu_host_alloc(size_t n);      /* pinned host memory (DMA at full PCIe rate), nullptr on failure */
__attribute__((weak)) void band_pair_gpu_host_free(void *p);              /* every slot's buffers for blocks up to edge^3 now (0 ok) */
#ifdef __cplusplus
}
#endif
