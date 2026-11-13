/* 2:4 (pair-wise 4:8 for e2m1) structured-sparse MX MMA on sm_120a: the raw-instruction probe (operand / metadata layout discovery
   against a host reference) and the throughput microbenchmark dense vs sparse (DESIGN.md 2026-11-13 "2:4 sparse fp4 convs").
   ptxas lowers mma.sp::ordered_metadata m16n8k128 kind::mxf4 / mxf4nvf4 block_scale to OMMA.SF.SP.168128 and m16n8k64
   kind::mxf8f6f4 to QMMA.SF.SP.16864. */
#include "lp_sp24.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200   /* ARCH=86 (LP_SW): no sparse MX MMA, the probe returns zeros */
#define SP_HW 1
#endif

/* variants (lp_sp24_mma_probe / bench): 0 dense m16n8k64 mxf4 2X, 1 sparse m16n8k128 mxf4 2X, 2 sparse m16n8k128 mxf4nvf4 4X (ue8m0),
   3 dense m16n8k32 mxf8f6f4 e4m3 1X, 4 sparse m16n8k64 mxf8f6f4 e4m3 1X */
template <int V> __device__ __forceinline__ void sp_mma(float *c, const unsigned *a, const unsigned *b, unsigned e, unsigned sa, unsigned sb) {
#ifdef SP_HW
    if constexpr (V == 0)
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
    else if constexpr (V == 1)
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, %13, {0, 0}, %14, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                       "r"(e), "r"(sa), "r"(sb));
    else if constexpr (V == 2)
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, %13, {0, 0}, %14, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                       "r"(e), "r"(sa), "r"(sb));
    else if constexpr (V == 3)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
    else
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, %13, {0, 0}, %14, {0, 0};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                       "r"(e), "r"(sa), "r"(sb));
#else
    (void)c; (void)a; (void)b; (void)e; (void)sa; (void)sb;
#endif
}

/* one warp, one MMA: lane l's registers from in[l][11] = a0..a3, b0..b3, e, sa, sb; out[l][4] = c0..c3 */
template <int V> __global__ void sp_probe_k(const unsigned *in, float *out) {
    const int l = threadIdx.x;
    const unsigned *r = in + l * 11;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    sp_mma<V>(c, r, r + 4, r[8], r[9], r[10]);
    for (int i = 0; i < 4; i++) out[l * 4 + i] = c[i];
}
/* throughput: each warp runs IT x 8 independent accumulator chains (no memory in the loop) */
template <int V> __global__ void __launch_bounds__(256) sp_bench_k(const unsigned *in, float *out, int it) {
    const int l = threadIdx.x & 31;
    unsigned a[4], b[4];
    for (int i = 0; i < 4; i++) { a[i] = in[l * 11 + i]; b[i] = in[l * 11 + 4 + i]; }
    const unsigned e = in[l * 11 + 8], sa = in[l * 11 + 9], sb = in[l * 11 + 10];
    float c[8][4] = {};
    for (int i = 0; i < it; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) sp_mma<V>(c[j], a, b, e, sa, sb);
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == 1234.5f) out[0] = s;   /* keeps the chains live */
}

extern "C" int lp_sp24_mma_probe(int v, const unsigned *h_in, float *h_out) {
    unsigned *d_in; float *d_out;
    if (cudaMalloc(&d_in, 32 * 11 * 4) != cudaSuccess || cudaMalloc(&d_out, 32 * 4 * 4) != cudaSuccess) return -1;
    cudaMemcpy(d_in, h_in, 32 * 11 * 4, cudaMemcpyHostToDevice);
    switch (v) {
    case 0: sp_probe_k<0><<<1, 32>>>(d_in, d_out); break;
    case 1: sp_probe_k<1><<<1, 32>>>(d_in, d_out); break;
    case 2: sp_probe_k<2><<<1, 32>>>(d_in, d_out); break;
    case 3: sp_probe_k<3><<<1, 32>>>(d_in, d_out); break;
    default: sp_probe_k<4><<<1, 32>>>(d_in, d_out); break;
    }
    cudaError_t err = cudaMemcpy(h_out, d_out, 32 * 4 * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    return err == cudaSuccess && cudaGetLastError() == cudaSuccess ? 0 : -1;
}
/* effective dense-equivalent TFLOP/s of variant v (sparse counts its logical K) */
extern "C" double lp_sp24_mma_bench(int v, const unsigned *h_in, int blocks, int it) {
    unsigned *d_in; float *d_out;
    if (cudaMalloc(&d_in, 32 * 11 * 4) != cudaSuccess || cudaMalloc(&d_out, 16) != cudaSuccess) return -1;
    cudaMemcpy(d_in, h_in, 32 * 11 * 4, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int rep = 0; rep < 4; rep++) {
        cudaEventRecord(e0);
        switch (v) {
        case 0: sp_bench_k<0><<<blocks, 256>>>(d_in, d_out, it); break;
        case 1: sp_bench_k<1><<<blocks, 256>>>(d_in, d_out, it); break;
        case 2: sp_bench_k<2><<<blocks, 256>>>(d_in, d_out, it); break;
        case 3: sp_bench_k<3><<<blocks, 256>>>(d_in, d_out, it); break;
        default: sp_bench_k<4><<<blocks, 256>>>(d_in, d_out, it); break;
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (rep && ms < best) best = ms;   /* rep 0: warm-up */
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(d_in); cudaFree(d_out);
    if (cudaGetLastError() != cudaSuccess) return -1;
    const double k = v == 0 ? 64 : v == 1 || v == 2 ? 128 : v == 3 ? 32 : 64;
    const double flop = 2.0 * 16 * 8 * k * 8 * (double)it * blocks * 8;   /* 8 warps per block, 8 chains */
    return flop / (best * 1e-3) * 1e-12;
}
