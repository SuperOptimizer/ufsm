/* ftgpu.h - the per-voxel fibre axis of ufsm fiber-trace on the CPU (src/ftaxis.c, the reference) and on the GPU (src/nn_ftrace.cu).
   Both write, per voxel of the n[0] x n[1] x n[2] presence volume pr, the axis in the published encoding (nx, ny: x, y of the unit axis
   on the z >= 0 hemisphere, 128 + 127 c) from the presence's gradient structure tensor summed over (2R+1)^3 voxels (prior 'H' / 'V':
   the family direction in the layer plane instead of the tube axis). The GPU path is byte-identical to the CPU path: the tensor sums
   are exact integers, and nn_ftrace.cu is built with -fmad=false and IEEE division / sqrt like gcc -std=c23 (no contraction). */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void ft_ridge_axis(const uint8_t *pr, const int64_t n[3], uint8_t *nx, uint8_t *ny, int R, char prior);
/* the same on device gpu; 0 on success, nonzero on any CUDA failure (message on stderr; the caller falls back to the CPU) */
int ft_ridge_axis_gpu(int gpu, const uint8_t *pr, const int64_t n[3], uint8_t *nx, uint8_t *ny, int R, char prior);
#ifdef __cplusplus
}
#endif
