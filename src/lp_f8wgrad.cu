/* FP8 3^3 stride-1 weight gradient, lp_check, bf16 helpers and the fp32 public entry points. */
#include "lp_f8wgrad.cuh"

extern "C" void lp_set_f8w_coop(int c) { g_f8w_coop = c; }
/* test probe: mean of n sr_e4m3x roundings of v (then the round-to-nearest conversion, as the kernels store them) */
__global__ void sr_e4m3x_probe_k(float v, size_t n, double *acc) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(acc, (double)dec_e4m3x2((unsigned short)cvt_e4m3(sr_e4m3x(v, 0x7654321u, i))).x);
}
extern "C" double lp_sr_e4m3x_mean(float v, size_t n) {
    double *d; cudaMalloc(&d, sizeof(double)); cudaMemset(d, 0, sizeof(double));
    sr_e4m3x_probe_k<<<nblk_(n, 256), 256>>>(v, n, d);
    double h = 0; cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d); LPCK();
    return h / (double)n;
}
int g_f8w_coop = 1 | 4;   /* gy (bit 1) measured mixed: -7% at 96->32, +5..14% elsewhere (per-channel work per thread); bit 2 (auto): gy
                             coop for the network input conv only (Ci <= 8, one 16-channel gy block: 16.7 -> 13.1 ms at 384^3) */
/* instantiated in lp_f8wgrad_i*.cu */
extern template void bwd_w_f8_t<mx4_t, mx8_t>(const void *x, shape5 xs, const mx8_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
extern template void bwd_w_f8_t<mx4_t, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<mx4_t, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<mx4_t, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
extern template void bwd_w_f8_t<mx8_t, mx8_t>(const void *x, shape5 xs, const mx8_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
extern template void bwd_w_f8_t<mx8_t, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<mx8_t, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<mx8_t, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
extern template void bwd_w_f8_t<__half, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<__half, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<bf16, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<bf16, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f8_t<float, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif

template void bwd_w_f8_t<mx8_t, mx4_t>(const void *x, shape5 xs, const mx4_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
template void bwd_w_f8_t<mx4_t, mx4_t>(const void *x, shape5 xs, const mx4_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
thread_local f8w_gnf_t g_f8w_gnf;
extern "C" int lp_gnpend_of(const void *gx);
extern "C" int lp_gnpend_flush_of(const void *gx);
extern "C" const void *lp_gnpend_take(const void *gx, const void **gy, const float **cf);
static int g_stem_gnfuse = -1;
extern "C" void lp_set_stem_gnfuse(int on) { g_stem_gnfuse = on; }
/* the network input conv's weight gradient takes over a pending GN+SiLU backward apply of its gy (unet: enc[0]'s GN1, whose gx only
   this weight gradient reads): the apply runs inside the FAST path's gy staging and gx is never written (saves the apply's pass: P512
   ~23 ms, and a fp4 rounding of the operand). Only for the FAST path's shape (<= 8 MX inputs, 32 MX-fp4 outputs, N 1, no z-window) */
static int stem_gnfuse(const void *x, int xbf, shape5 xs, const void *gy, shape5 ys, split_t sp) {
    if (g_stem_gnfuse < 0) g_stem_gnfuse = getenv("UFSM_STEM_GNFUSE") ? atoi(getenv("UFSM_STEM_GNFUSE")) : 1;
    int a, c;
    if (!g_stem_gnfuse || !lp_gnpend_of(gy) || (xbf != 3 && xbf != 4) || !x || xs.c > 8 || ys.c != 32 || ys.n != 1 || ys.w % 4 || sp.up || sp.x2 || zw_win(ys.d, &a, &c)) return 0;
    if ((g_f8w_coop & 3) != 3 && !((g_f8w_coop & 1) && (g_f8w_coop & 4))) return 0;   /* (launch_bw8's coop: the FAST path) */
    const void *g2 = nullptr; const float *cf = nullptr;
    const void *ax = lp_gnpend_take(gy, &g2, &cf);
    if (!ax || g2 != gy) return 0;
    g_f8w_gnf = f8w_gnf_t{(const uint8_t *)ax, (const float4 *)cf};
    return 1;
}
extern "C" int lp_bwd_w_f8(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    if (gybf == 4) {   /* MX-fp4 gradients: the network input conv (FAST path) only */
        const int fuse = stem_gnfuse(x, xbf, xs, gy, ys, sp);
        if (!fuse) lp_gnpend_flush_of(gy);
        if (xbf == 3) bwd_w_f8_t<mx8_t, mx4_t>(x, xs, (const mx4_t *)gy, ys, gw, gb, gp, sp);
        else if (xbf == 4) bwd_w_f8_t<mx4_t, mx4_t>(x, xs, (const mx4_t *)gy, ys, gw, gb, gp, sp);
        else { fprintf(stderr, "lp_bwd_w_f8: MX-fp4 gradients need an MX x\n"); abort(); }
        g_f8w_gnf = f8w_gnf_t{nullptr, nullptr};
        LPCK();
        return 0;
    }
    lp_gnpend_flush_of(gy);
    if (xbf == 4 && gybf == 3) bwd_w_f8_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 4 && gybf == 2) bwd_w_f8_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 4 && gybf == 1) VERIFY_ONLY(bwd_w_f8_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp));
    else if (xbf == 4) VERIFY_ONLY(bwd_w_f8_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp));
    else if (xbf == 3 && gybf == 3) bwd_w_f8_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3 && gybf == 2) bwd_w_f8_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 3 && gybf == 1) VERIFY_ONLY(bwd_w_f8_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp));
    else if (xbf == 3) VERIFY_ONLY(bwd_w_f8_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp));
    else if (xbf == 2 && gybf) bwd_w_f8_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp);
    else if (xbf == 2) VERIFY_ONLY(bwd_w_f8_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp));
    else if (xbf && gybf) VERIFY_ONLY(bwd_w_f8_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp));
    else if (xbf) VERIFY_ONLY(bwd_w_f8_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp));
    else if (gybf) { fprintf(stderr, "lp_bwd_w_f8: bf16 gradient with fp32 activations is not supported\n"); abort(); }
    else VERIFY_ONLY(bwd_w_f8_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp));
    LPCK();
    return 0;
}

extern "C" const char *lp_check(void) { cudaError_t e = g_lp_err; g_lp_err = cudaSuccess; return e == cudaSuccess ? nullptr : cudaGetErrorString(e); }
/* fp32 -> bf16 copy (test helper for the bf16-activation instantiations) */
__global__ void lp_f2bf_k(const float *x, bf16 *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]); }
__global__ void lp_bf2f_k(const bf16 *x, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = __bfloat162float(x[i]); }
extern "C" void lp_f32_to_bf16(const float *x, size_t n, void *y) { lp_f2bf_k<<<nblk_(n, 256), 256>>>(x, (bf16 *)y, n); LPCK(); }
extern "C" void lp_bf16_to_f32(const void *x, size_t n, float *y) { lp_bf2f_k<<<nblk_(n, 256), 256>>>((const bf16 *)x, y, n); LPCK(); }

/* ---- public direct entry points (fp32 tensors) ---- */
extern "C" void nn_conv3d_fwd_fp8(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_conv_fwd_f8(x, 0, xs, w, b, cout, y, 0, none, nullptr, 0, ns);
}
extern "C" void nn_conv3d_fwd_fp4(const float *x, shape5 xs, const float *w, const float *b, int cout, float *y) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_conv_fwd_f4(x, 0, xs, w, b, cout, y, 0, none, nullptr, 0, ns);
}
extern "C" void nn_conv3d_bwd_weight_fp8(const float *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb) {
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, 0};
    lp_bwd_w_f8(x, 0, xs, gy, 0, ys, gw, gb, none, ns);
}
