/* The mesh codes stamp on the GPU (src/nn_mcodes.cu): meshcodes.c mc_stamp's samples, byte for byte. The CPU rule is order defined (a voxel
   takes the first sample's code, in (tile, row, column, a, b) order, and becomes 255 when a later sample is more than 4 codes from it), so
   the device keeps, per voxel of a z slab, the 64-bit minimum of (order key << 8 | code) (pass 1), then every sample compares its code with
   the winner's and sets bit 63 on a conflict (pass 2): the first sample and "any sample > 4 from it" are what the CPU's sequence computes.
   Double arithmetic with no FMA contraction (-fmad=false), IEEE sqrt / division, llround: the samples land in the CPU's voxels with its
   codes. The host gathers the vertex blocks of the tiles that reach the box (meshcodes.c); per calling thread a stream and its buffers.
   Declared weak: without nn_mcodes.o the CPU stamp runs. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* one tile of the read: its vertex block (rows r0..r1, columns c0..c1 inclusive: nr x nc vertices, row major) starts at vertex voff of
   the gathered arrays; zlo / zhi its level-0 z bounds (the slab cull) */
typedef struct { uint32_t voff; int32_t nr, nc; float zlo, zhi; } mcg_tile;
/* the box o / n (level voxels, C order z, y, x) into out (host, n bytes): 0 ok, nonzero a CUDA failure or a box beyond the key's range
   (the caller runs the CPU stamp). xyz: level-0 x, y, z per vertex; sc = 2^-level. */
__attribute__((weak)) int mesh_codes_gpu_stamp(const mcg_tile *t, int nt, const float *xyz, const float *val, const uint8_t *valid, size_t nv, double sc,
                                               const int64_t o[3], const int64_t n[3], uint8_t *out);
__attribute__((weak)) void mesh_codes_gpu_device(int dev);
#ifdef __cplusplus
}
#endif
