/* MX tensor conversions and rounding probes */
#include "lp_mxops.cuh"
/* Elementwise, 1x1, GroupNorm and upsample ops on MX activation tensors. */
#include "lp_common.cuh"

extern "C" size_t lp_mx8_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) + 1); }
extern "C" size_t lp_mx4_bytes(int N, int C, size_t S) { return (size_t)N * mx_nb(C) * S * (mx_bw(C) / 2 + 1); }

template <int B, typename TI>
__global__ void f32_to_mx_k(const TI *x, uint8_t *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; r[k] = k < bw && c < C ? ldx(x, ((size_t)n * C + c) * S + v) : 0.f; }
    mx_store_row_b<B>(y, mx_sc<B>(y, N, C, S), i, bw, r);
}
extern "C" void lp_f32_to_mx8(const float *x, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    f32_to_mx_k<8, float><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, N, C, S); LPCK();
}
extern "C" void lp_f32_to_mx4(const float *x, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    f32_to_mx_k<4, float><<<nblk_(n, 256), 256>>>(x, (uint8_t *)y, N, C, S); LPCK();
}
/* 16-bit (dt 1 bf16, 2 fp16) [n][C][S] -> MX */
extern "C" void lp_h16_to_mx8(const void *x, int dt, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    if (dt == 2) f32_to_mx_k<8, __half><<<nblk_(n, 256), 256>>>((const __half *)x, (uint8_t *)y, N, C, S);
    else f32_to_mx_k<8, bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, (uint8_t *)y, N, C, S);
    LPCK();
}
extern "C" void lp_h16_to_mx4(const void *x, int dt, int N, int C, size_t S, void *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    if (dt == 2) f32_to_mx_k<4, __half><<<nblk_(n, 256), 256>>>((const __half *)x, (uint8_t *)y, N, C, S);
    else f32_to_mx_k<4, bf16><<<nblk_(n, 256), 256>>>((const bf16 *)x, (uint8_t *)y, N, C, S);
    LPCK();
}
/* MX -> fp32 [n][C][S] (tests) */
template <int B>
__global__ void mx_to_f32_k(const uint8_t *x, float *y, int N, int C, size_t S) {
    const int bw = mx_bw(C), nb = mx_nb(C);
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nb * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nb), n = (int)(i / (S * nb));
    float r[32];
    mx_load_row_b<B>(x, mx_sc<B>(x, N, C, S), i, bw, r);
#pragma unroll
    for (int k = 0; k < 32; k++) { int c = blk * bw + k; if (k < bw && c < C) y[((size_t)n * C + c) * S + v] = r[k]; }
}
extern "C" void lp_mx8_to_f32(const void *x, int N, int C, size_t S, float *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    mx_to_f32_k<8><<<nblk_(n, 256), 256>>>((const uint8_t *)x, y, N, C, S); LPCK();
}
extern "C" void lp_mx4_to_f32(const void *x, int N, int C, size_t S, float *y) {
    size_t n = (size_t)N * mx_nb(C) * S;
    mx_to_f32_k<4><<<nblk_(n, 256), 256>>>((const uint8_t *)x, y, N, C, S); LPCK();
}
/* test probes: mean of n stochastic e2m1 roundings of v (seeded per element), and the raw cvt.rn.satfinite.e2m1x2 nibble of v */
__global__ void sr_e2m1_probe_k(float v, size_t n, double *acc) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q = sr_e2m1(v, sr_hash(0x1234567u, i));
    atomicAdd(acc, (double)q);
}
extern "C" double lp_sr_e2m1_mean(float v, size_t n) {
    double *d; cudaMalloc(&d, sizeof(double)); cudaMemset(d, 0, sizeof(double));
    sr_e2m1_probe_k<<<nblk_(n, 256), 256>>>(v, n, d);
    double h = 0; cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d); LPCK();
    return h / (double)n;
}
__global__ void cvt_e2m1_probe_k(const float *v, unsigned char *o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = cvt_e2m1x2(v[i], 0.f) & 15; }
extern "C" void lp_cvt_e2m1_probe(const float *hv, unsigned char *ho, int n) {
    float *d; unsigned char *o; cudaMalloc(&d, n * 4); cudaMalloc(&o, n); cudaMemcpy(d, hv, n * 4, cudaMemcpyHostToDevice);
    cvt_e2m1_probe_k<<<nblk_(n, 128), 128>>>(d, o, n);
    cudaMemcpy(ho, o, n, cudaMemcpyDeviceToHost); cudaFree(d); cudaFree(o); LPCK();
}
/* elementwise ops on MX activations of either format: B / BX / BY = element bits of the MX operand (8: mx8, 4: mx4), BX = 0 a
   plane-major input of type TI. lp dtype codes at the entry points: 0 fp32, 1 bf16, 2 fp16, 3 mx8, 4 mx4. */
