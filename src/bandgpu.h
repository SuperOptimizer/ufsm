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
/* One stored 128^3 codes chunk for band_pair_gpu_codes (volcomp lossless, src/band_onfly.c packs them): o its origin in the box;
   mode 1 (coded: 8 tANS tables of 1024 packed entries ns | sym << 16 | nb << 24 at tab_off, substream s's token / bypass bytes at
   tok_off / byp_off, 4-byte aligned, in the blob), 2 (raw: 128^3 bytes at raw_off) or 3 (const value). */
typedef struct bg_lchunk { int32_t o[3]; uint32_t mode, value, tab_off, raw_off, pad; uint32_t tok_off[32], tok_n[32], byp_off[32], byp_n[32]; } bg_lchunk;
/* the pair rule on a box of n voxels whose codes are decoded on the device from the chunks (voxels outside lo .. hi, the codes' shape
   within the box, and outside every chunk are 0); the phase of the crop box (crop_o, crop_n within the box) into out (crop_n bytes,
   C order); rmax <= 0: the decoded codes of the crop instead (tests). 0 ok; 1 a chunk did not decode on the GPU (decode on the CPU: the reference's status); -1 a CUDA failure. */
__attribute__((weak)) int band_pair_gpu_codes(const bg_lchunk *ch, int nch, const uint8_t *blob, size_t blob_n, const int n[3], const int lo[3], const int hi[3],
                                              float rmax, const int crop_o[3], const int crop_n[3], uint8_t *out);
__attribute__((weak)) int band_pair_gpu(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase);
__attribute__((weak)) void band_pair_gpu_slots(int slots, int device);   /* before the first call: concurrent callers, the device */
__attribute__((weak)) void band_pair_gpu_free(void);                     /* the idle slots' buffers */
__attribute__((weak)) int band_pair_gpu_reserve(int edge);
__attribute__((weak)) void *band_pair_gpu_host_alloc(size_t n);      /* pinned host memory (DMA at full PCIe rate), nullptr on failure */
__attribute__((weak)) void band_pair_gpu_host_free(void *p);              /* every slot's buffers for blocks up to edge^3 now (0 ok) */
#ifdef __cplusplus
}
#endif
