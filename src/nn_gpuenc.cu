/* volcomp lossy encoding on the GPU, the analysis half (src/vcgpu.h): every 16^3 block of a uint8 shard gathered, transformed and
   quantised with the CPU encoder's arithmetic (volcomp.h vf_analyse_block: x, y, z passes of VF_DCT16_FWD_BODY with every operation
   rounded (this file is built with -fmad=false and no fast math), the DC rounded unfused, the AC dead-zone quantiser fused as the AVX2
   kernel's _mm256_fmadd_ps or unfused as the C kernel's), then each block's nonzero AC levels listed in ascending scan order. Three
   passes: count (vcg_count_k), the exclusive scan of the counts (vcg_scan_k), list (vcg_list_k, the block recomputed: cheaper than
   holding 16 KB of levels a block). The layout of the output buffer: vcg_layout (vcgpu.h). */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "vcgpu.h"

/* volcomp.h's 16-point forward transform (copied verbatim; tests/test_vcgpu checks the streams against volcomp_encode) */
#define VCG_DCT16_FWD_BODY(VT) \
  VT t1 = in[0] + in[15]; \
  VT t2 = in[1] + in[14]; \
  VT t3 = in[2] + in[13]; \
  VT t4 = in[3] + in[12]; \
  VT t5 = in[4] + in[11]; \
  VT t6 = in[5] + in[10]; \
  VT t7 = in[6] + in[9]; \
  VT t8 = in[7] + in[8]; \
  VT t9 = (in[0] - in[15]) * 5.024192862e-01f; \
  VT t10 = (in[1] - in[14]) * 5.224986149e-01f; \
  VT t11 = (in[2] - in[13]) * 5.669440348e-01f; \
  VT t12 = (in[3] - in[12]) * 6.468217834e-01f; \
  VT t13 = (in[4] - in[11]) * 7.881546235e-01f; \
  VT t14 = (in[5] - in[10]) * 1.060677686e+00f; \
  VT t15 = (in[6] - in[9]) * 1.722447098e+00f; \
  VT t16 = (in[7] - in[8]) * 5.101148619e+00f; \
  VT t17 = t1 + t8; \
  VT t18 = t2 + t7; \
  VT t19 = t3 + t6; \
  VT t20 = t4 + t5; \
  VT t21 = (t1 - t8) * 5.097955791e-01f; \
  VT t22 = (t2 - t7) * 6.013448869e-01f; \
  VT t23 = (t3 - t6) * 8.999762231e-01f; \
  VT t24 = (t4 - t5) * 2.562915448e+00f; \
  VT t25 = t17 + t20; \
  VT t26 = t18 + t19; \
  VT t27 = (t17 - t20) * 5.411961001e-01f; \
  VT t28 = (t18 - t19) * 1.306562965e+00f; \
  VT t29 = t25 + t26; \
  VT t30 = (t25 - t26) * 7.071067812e-01f; \
  VT t31 = t27 + t28; \
  VT t32 = (t27 - t28) * 7.071067812e-01f; \
  VT t33 = t31 + t32; \
  VT t34 = t21 + t24; \
  VT t35 = t22 + t23; \
  VT t36 = (t21 - t24) * 5.411961001e-01f; \
  VT t37 = (t22 - t23) * 1.306562965e+00f; \
  VT t38 = t34 + t35; \
  VT t39 = (t34 - t35) * 7.071067812e-01f; \
  VT t40 = t36 + t37; \
  VT t41 = (t36 - t37) * 7.071067812e-01f; \
  VT t42 = t40 + t41; \
  VT t43 = t38 + t42; \
  VT t44 = t42 + t39; \
  VT t45 = t39 + t41; \
  VT t46 = t9 + t16; \
  VT t47 = t10 + t15; \
  VT t48 = t11 + t14; \
  VT t49 = t12 + t13; \
  VT t50 = (t9 - t16) * 5.097955791e-01f; \
  VT t51 = (t10 - t15) * 6.013448869e-01f; \
  VT t52 = (t11 - t14) * 8.999762231e-01f; \
  VT t53 = (t12 - t13) * 2.562915448e+00f; \
  VT t54 = t46 + t49; \
  VT t55 = t47 + t48; \
  VT t56 = (t46 - t49) * 5.411961001e-01f; \
  VT t57 = (t47 - t48) * 1.306562965e+00f; \
  VT t58 = t54 + t55; \
  VT t59 = (t54 - t55) * 7.071067812e-01f; \
  VT t60 = t56 + t57; \
  VT t61 = (t56 - t57) * 7.071067812e-01f; \
  VT t62 = t60 + t61; \
  VT t63 = t50 + t53; \
  VT t64 = t51 + t52; \
  VT t65 = (t50 - t53) * 5.411961001e-01f; \
  VT t66 = (t51 - t52) * 1.306562965e+00f; \
  VT t67 = t63 + t64; \
  VT t68 = (t63 - t64) * 7.071067812e-01f; \
  VT t69 = t65 + t66; \
  VT t70 = (t65 - t66) * 7.071067812e-01f; \
  VT t71 = t69 + t70; \
  VT t72 = t67 + t71; \
  VT t73 = t71 + t68; \
  VT t74 = t68 + t70; \
  VT t75 = t58 + t72; \
  VT t76 = t72 + t62; \
  VT t77 = t62 + t73; \
  VT t78 = t73 + t59; \
  VT t79 = t59 + t74; \
  VT t80 = t74 + t61; \
  VT t81 = t61 + t70; \
  out[0] = t29; \
  out[1] = t75; \
  out[2] = t43; \
  out[3] = t76; \
  out[4] = t33; \
  out[5] = t77; \
  out[6] = t44; \
  out[7] = t78; \
  out[8] = t30; \
  out[9] = t79; \
  out[10] = t45; \
  out[11] = t80; \
  out[12] = t32; \
  out[13] = t81; \
  out[14] = t41; \
  out[15] = t70;

__device__ static inline void vcg_fwd16(const float in[16], float out[16]) { VCG_DCT16_FWD_BODY(float) }

/* block g of the n^3 shard (chunk-major, chunks z-major, then bi = bz * 64 + by * 8 + bx): gathered into blk (shared, z-major 16^3),
   transformed unless constant; returns the constant value (0..255) or -1 */
__device__ static int vcg_block(const uint8_t *__restrict__ src, int n, long g, float *blk, int *smm) {
    const int t = threadIdx.x, m = n / 128, c = (int)(g / 512), bi = (int)(g % 512);
    const int cz = c / (m * m), cy = c / m % m, cx = c % m, bz = bi >> 6, by = bi >> 3 & 7, bx = bi & 7;
    if (t == 0) { smm[0] = 255; smm[1] = 0; }
    __syncthreads();
    {   const int z = t >> 4, y = t & 15;
        const uint8_t *row = src + ((size_t)(cz * 128 + bz * 16 + z) * n + (cy * 128 + by * 16 + y)) * n + cx * 128 + bx * 16;
        int mn = 255, mx = 0;
        for (int x = 0; x < 16; x++) { const int v = row[x]; mn = min(mn, v); mx = max(mx, v); blk[z * 256 + y * 16 + x] = __fsub_rn((float)v, 128.0f); }
        atomicMin(&smm[0], mn); atomicMax(&smm[1], mx); }
    __syncthreads();
    if (smm[0] == smm[1]) return smm[0];
    float in[16], out[16];
    {   const int z = t >> 4, y = t & 15; float *l = blk + z * 256 + y * 16;   /* x lines */
        for (int k = 0; k < 16; k++) in[k] = l[k];
        vcg_fwd16(in, out); for (int k = 0; k < 16; k++) l[k] = out[k]; }
    __syncthreads();
    {   const int z = t >> 4, x = t & 15; float *l = blk + z * 256 + x;   /* y lines */
        for (int k = 0; k < 16; k++) in[k] = l[k * 16];
        vcg_fwd16(in, out); for (int k = 0; k < 16; k++) l[k * 16] = out[k]; }
    __syncthreads();
    {   const int i = t; float *l = blk + i;   /* z lines */
        for (int k = 0; k < 16; k++) in[k] = l[k * 256];
        vcg_fwd16(in, out); for (int k = 0; k < 16; k++) l[k * 256] = out[k]; }
    __syncthreads();
    return -1;
}
__device__ static inline int vcg_q(float v, float rs, int fma) {   /* the AC dead-zone quantiser (VF_DZ_Q 0.2) */
    const float a = fma ? __fmaf_rn(fabsf(v), rs, 0.2f) : __fadd_rn(__fmul_rn(fabsf(v), rs), 0.2f);
    const int l = (int)a; return v < 0.0f ? -l : l;
}
__device__ static inline int vcg_dc(float c0, float rs0) {   /* DC: round to nearest (VF_DZ_Q_DC 0.5), unfused */
    const int l = (int)__fadd_rn(__fmul_rn(fabsf(c0), rs0), 0.5f); return c0 < 0.0f ? -l : l;
}
__global__ static void vcg_count_k(const uint8_t *__restrict__ src, int n, const float *__restrict__ rstep, int fma, int32_t *dc, uint32_t *cnt, int16_t *cv) {
    __shared__ float blk[4096]; __shared__ int smm[2], sc;
    const long g = blockIdx.x; const int t = threadIdx.x;
    const int k = vcg_block(src, n, g, blk, smm);
    if (k >= 0) { if (t == 0) { dc[g] = vcg_dc(__fmul_rn(__fsub_rn((float)k, 128.0f), 4096.0f), rstep[0]); cnt[g] = 0; cv[g] = (int16_t)k; } return; }
    if (t == 0) sc = 0;
    __syncthreads();
    int c = 0;
    for (int i = t; i < 4096; i += 256) c += i != 0 && vcg_q(blk[i], rstep[i], fma) != 0;
    atomicAdd(&sc, c);
    __syncthreads();
    if (t == 0) { dc[g] = vcg_dc(blk[0], rstep[0]); cnt[g] = (uint32_t)sc; cv[g] = -1; }
}
/* exclusive scan of cnt[0..nb) into off, the total into *tot (one block of 1024 threads) */
__global__ static void vcg_scan_k(const uint32_t *__restrict__ cnt, long nb, uint32_t *off, unsigned long long *tot) {
    __shared__ unsigned long long s[1024];
    const int t = threadIdx.x; const long per = (nb + 1023) / 1024, a = t * per, b = a + per < nb ? a + per : nb;
    unsigned long long sum = 0; for (long i = a; i < b; i++) sum += cnt[i];
    s[t] = sum; __syncthreads();
    for (int d = 1; d < 1024; d <<= 1) { const unsigned long long v = t >= d ? s[t - d] : 0; __syncthreads(); s[t] += v; __syncthreads(); }
    unsigned long long run = s[t] - sum;
    for (long i = a; i < b; i++) { off[i] = (uint32_t)run; run += cnt[i]; }
    if (t == 1023) *tot = s[1023];
}
/* the levels of a block staged in shared memory, then written as one contiguous run (lv / pos may be pinned host memory: the writes cross
   the bus in whole lines; scattered ones took 4x longer at q 2) */
__global__ static void vcg_list_k(const uint8_t *__restrict__ src, int n, const float *__restrict__ rstep, const uint16_t *__restrict__ scan, int fma,
                                  const uint32_t *__restrict__ off, const uint32_t *__restrict__ cnt, const unsigned long long *__restrict__ tot, unsigned long long cap, int32_t *lv, uint16_t *pos) {
    __shared__ float blk[4096]; __shared__ int smm[2]; __shared__ int ss[256]; __shared__ int32_t sl[4096]; __shared__ uint16_t sp[4096];
    if (*tot > cap) return;   /* (the host encodes this shard from its bytes) */
    const long g = blockIdx.x; const int t = threadIdx.x;
    if (!cnt[g]) return;   /* (constant blocks, and blocks with a DC only) */
    if (vcg_block(src, n, g, blk, smm) >= 0) return;
    int q[16], c = 0;   /* scan positions 16 t .. 16 t + 15 */
    for (int j = 0; j < 16; j++) { const int p = t * 16 + j, nat = scan[p]; q[j] = nat ? vcg_q(blk[nat], rstep[nat], fma) : 0; c += q[j] != 0; }
    ss[t] = c; __syncthreads();
    for (int d = 1; d < 256; d <<= 1) { const int v = t >= d ? ss[t - d] : 0; __syncthreads(); ss[t] += v; __syncthreads(); }
    int o = ss[t] - c;
    for (int j = 0; j < 16; j++) if (q[j]) { sl[o] = q[j]; sp[o] = (uint16_t)(t * 16 + j); o++; }
    __syncthreads();
    const int m = ss[255]; const size_t b = off[g];
    for (int i = t; i < m; i += 256) { lv[b + i] = sl[i]; pos[b + i] = sp[i]; }
}
extern "C" int nn_vcg_shard(const uint8_t *dsh, int n, const float *drstep, const uint16_t *dscan, int fma, uint8_t *hdr, int32_t *lv, uint16_t *pos, size_t cap_e) {
    if (n % 128) return -1;
    vcg_layout L; vcg_layout_of(&L, n, cap_e);
    int32_t *dc = (int32_t *)(hdr + L.o_dc); uint32_t *cnt = (uint32_t *)(hdr + L.o_cnt), *off = (uint32_t *)(hdr + L.o_off);
    int16_t *cv = (int16_t *)(hdr + L.o_cv); unsigned long long *tot = (unsigned long long *)hdr;
    vcg_count_k<<<(unsigned)L.nb, 256>>>(dsh, n, drstep, fma, dc, cnt, cv);
    vcg_scan_k<<<1, 1024>>>(cnt, L.nb, off, tot);
    vcg_list_k<<<(unsigned)L.nb, 256>>>(dsh, n, drstep, dscan, fma, off, cnt, tot, cap_e, lv, pos);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* the raw shard into pinned host memory (written over the bus by the kernel) only when its levels overflowed (*tot > cap): the writer then
   encodes it from the bytes; otherwise nothing crosses the bus (the heads' shards were copied whole beside their levels: 92 GB of a 5-head
   19 Gvox run, ahead of the --fit-zarr copies on the offload stream) */
__global__ static void vcg_copy_over_k(const uint4 *__restrict__ src, uint4 *dst, size_t n16, const unsigned long long *__restrict__ tot, unsigned long long cap) {
    if (*tot <= cap) return;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
extern "C" int nn_vcg_copy_over(const uint8_t *dsh, uint8_t *host, size_t bytes, const uint8_t *hdr, size_t cap_e) {
    if (bytes % 16 || ((uintptr_t)dsh | (uintptr_t)host) % 16) return -1;
    vcg_copy_over_k<<<1024, 256>>>((const uint4 *)dsh, (uint4 *)host, bytes / 16, (const unsigned long long *)hdr, cap_e);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
