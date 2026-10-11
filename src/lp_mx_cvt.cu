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
/* the network input of predict (nn_pred_input: z-scored CT, 0, and the unit direction away from the axis) for window planes
   [z0, z0 + nz), each value rounded to the 16-bit input type (HD 2 fp16, 1 bf16, 0 none) and stored as MX-fp8 rows of one
   8-channel block (channels 4..7 zero): bit for bit nn_pred_input followed by the 16-bit -> MX-fp8 input conversion, without the
   16-bit window. y: an MX-fp8 tensor of 4 channels and depth nz. */
template <int HD> __device__ __forceinline__ float in_round(float v) {
    if constexpr (HD == 2) return __half2float(__float2half(v));
    else if constexpr (HD == 1) return __bfloat162float(__float2bfloat16(v));
    else return v;
}
template <int HD>
__global__ void pred_in_mx8_k(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, uint8_t *y) {
    const size_t S = (size_t)nz * W * W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= S) return;
    const int xx = (int)(i % W), yy = (int)((i / W) % W), z = z0 + (int)(i / ((size_t)W * W));
    const float dy = dyo[z] + (float)yy, dx = dxo[z] + (float)xx, inv = axis ? (float)axis / (sqrtf(dy * dy + dx * dx) + 1e-6f) : 0.f;
    float r[32];
#pragma unroll
    for (int k = 0; k < 32; k++) r[k] = 0.f;
    r[0] = in_round<HD>(((float)ct[(size_t)z * W * W + (size_t)yy * W + xx] - mean) * isd); r[1] = in_round<HD>(0.f);
    r[2] = in_round<HD>(dy * inv); r[3] = in_round<HD>(dx * inv);
    mx_store_row_b<8>(y, mx_sc<8>(y, 1, 4, S), i, 8, r);
}
extern "C" void lp_pred_input_mx8(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, int z0, int nz, void *y, int hd) {
    const size_t S = (size_t)nz * W * W;
    if (hd == 2) pred_in_mx8_k<2><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, (uint8_t *)y);
    else if (hd == 1) pred_in_mx8_k<1><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, (uint8_t *)y);
    else pred_in_mx8_k<0><<<nblk_(S, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, z0, nz, (uint8_t *)y);
    LPCK();
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
