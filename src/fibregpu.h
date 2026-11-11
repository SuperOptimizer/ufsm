/* the fibre planes of one box on the GPU (src/nn_fibre.cu; src/fiberraster.c fr_box's CPU stamp and combine, byte for byte: double
   arithmetic with no FMA contraction (-fmad=false, __dmul_rn / __dadd_rn), IEEE sqrt, the same cube bounds). Declared weak: without
   nn_fibre.o the CPU code runs. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { double p[3]; int32_t fam, pad; } fr_step;   /* a polyline step (box-level voxels, z y x); fam 0 H, 1 V, 2 other, 3 an --extra veto */
/* steps (any order); so / n the box (level voxels); R the reach of fam 0..2, Rx of fam 3; cr / cg = (int)(16 radius), (int)(16 (radius + guard));
   ring > 0: the negative ring; ct: the box's CT (n voxels, 255 outside the CT) or nullptr (no air test), iso the cutoff. o0 / o1: the H / V
   planes (n voxels, host). 0 ok, nonzero on a CUDA failure (the caller runs the CPU code). */
__attribute__((weak)) int fibre_gpu_box(const fr_step *st, long nst, const int64_t so[3], const int64_t n[3], double R, double Rx, int cr, int cg, int ring,
                                        const uint8_t *ct, int iso, uint8_t *o0, uint8_t *o1);
__attribute__((weak)) void fibre_gpu_device(int dev);
__attribute__((weak)) int fibre_gpu_reserve(int edge);   /* the device buffers for boxes up to edge^3 now (0 ok) */
__attribute__((weak)) void *fibre_gpu_host_alloc(size_t n);   /* pinned (the planes' D2H at full rate), nullptr on failure */
__attribute__((weak)) void fibre_gpu_host_free(void *p);   /* before the first call (default 0) */
#ifdef __cplusplus
}
#endif
