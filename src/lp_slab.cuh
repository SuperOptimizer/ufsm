/* Slab-local GroupNorm helpers for the readers of a normalised tensor (gnp_t ns > 0, nn_lp.h): statistics per (sample, z-slab). */
#pragma once
#include "nn_lp.h"
#include <cuda_runtime.h>
#include <limits.h>
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
/* a reader staging the stored planes [z0, z1] through a table of GN+SiLU coefficients: the first plane in (z0, z1] whose slab
   differs from z0's (planes < 0 count as plane 0), INT_MAX when the range is one slab or the statistics are per sample. Bank 0 of
   the table then holds z0's slab, bank 1 the slab of the planes >= zb. a: the gnp_t of the x segment, b: of x2 (split; a conv's
   segments share one slab geometry: the first with slabs decides). Two banks serve ranges of at most two slabs (gn_zb3). */
__device__ __forceinline__ int gn_zb(const gnp_t &a, const gnp_t &b, int z0, int z1) {
    const gnp_t &q = a.ns ? a : b;
    if (!q.ns) return INT_MAX;
    const int s = gn_row(q, 0, z0 < 0 ? 0 : z0);
    if (s >= q.ns - 1) return INT_MAX;
    const int zb = (s + 1) * q.zs - q.zo;
    return zb <= z1 ? zb : INT_MAX;
}
__device__ __forceinline__ int gn_zb(const gnp_t &a, int z0, int z1) { return gn_zb(a, a, z0, z1); }
/* a second coefficient bank of nf float2 at the end of the dynamic shared memory (launchers add nf * 8 + 16 bytes for slab GroupNorm) */
__device__ __forceinline__ float2 *gn_bank2(unsigned char *smem_raw, int nf) {
    unsigned dyn; asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
    return (float2 *)(smem_raw + ((dyn - (unsigned)nf * 8u) & ~15u));
}
/* the range [z0, z1] reaches a third slab (slabs thinner than the staged depth: the readers trap) */
__device__ __forceinline__ bool gn_zb3(const gnp_t &a, const gnp_t &b, int z0, int z1) {
    const int zb = gn_zb(a, b, z0, z1);
    return zb != INT_MAX && gn_zb(a, b, zb, z1) != INT_MAX;
}
