/* Slab-local GroupNorm helpers for the readers of a normalised tensor (gnp_t ns > 0, nn_lp.h): statistics per (sample, z-slab). */
#pragma once
#include "nn_lp.h"
#include <cuda_runtime.h>
/* the statistics row of sample n at plane z (n itself when ns == 0) */
__device__ __forceinline__ int gn_row(const gnp_t &gp, int n, int z) {
    if (!gp.ns) return n;
    int s = (z + gp.zo) / gp.zs;
    s = s < 0 ? 0 : s >= gp.ns ? gp.ns - 1 : s;
    return n * gp.ns + s;
}
__device__ __forceinline__ int gn_row_v(const gnp_t &gp, int n, size_t v) { return gp.ns ? gn_row(gp, n, (int)(v / (size_t)gp.zhw)) : n; }   /* voxel v of a plane-major sample */
/* the GN+SiLU affine (a, b) of channel ci (of Ci channels) for statistics row nr: the arithmetic of row_gn / make_chan */
__device__ __forceinline__ float2 gn_ab(const gnp_t &gp, int nr, int ci, int Ci) {
    const int ng = nr * gp.G + ci / (Ci / gp.G); const float a = gp.rstd[ng] * gp.gamma[ci];
    return make_float2(a, gp.beta[ci] - gp.mean[ng] * a);
}
