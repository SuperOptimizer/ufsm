/* FP4 3^3 stride-1 weight gradient. */
#include "lp_f4wgrad.cuh"

extern "C" void lp_set_f4w_coop(int on) { g_f4w_coop = on != 0; }
int g_f4w_coop = -1;
int g_f4w_gypre_kb = -1;   /* gy pre-pass slab cap in KiB (0 off, -1 env UFSM_F4W_GYPRE in MiB, default 96) */
extern "C" void lp_set_f4w_gypre_kb(int kb) { g_f4w_gypre_kb = kb; }
/* instantiated in lp_f4wgrad_i*.cu */
extern template void bwd_w_f4_t<mx4_t, mx8_t>(const void *x, shape5 xs, const mx8_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
extern template void bwd_w_f4_t<mx4_t, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<mx4_t, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<mx4_t, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
extern template void bwd_w_f4_t<mx8_t, mx8_t>(const void *x, shape5 xs, const mx8_t *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
extern template void bwd_w_f4_t<mx8_t, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<mx8_t, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<mx8_t, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
extern template void bwd_w_f4_t<__half, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<__half, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<bf16, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<bf16, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
#ifdef UFSM_ALL_TYPES
extern template void bwd_w_f4_t<float, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif

int g_f4w_gy4 = 0;
extern "C" int lp_bwd_w_f4(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had) {
    if (gybf == 4) {   /* MX-fp4 gradients: the gy pre-pass decodes the fp4 rows; the kernels read only its records */
        if (xbf != 4 && xbf != 3) { fprintf(stderr, "lp_bwd_w_f4: MX-fp4 gradients need an MX x\n"); abort(); }
        g_f4w_gy4 = 1;
        if (xbf == 4) bwd_w_f4_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
        else bwd_w_f4_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
        g_f4w_gy4 = 0;
        LPCK();
        return 0;
    }
    if (xbf == 4 && gybf == 3) bwd_w_f4_t<mx4_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 4 && gybf == 2) bwd_w_f4_t<mx4_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 4 && gybf == 1) VERIFY_ONLY(bwd_w_f4_t<mx4_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf == 4) VERIFY_ONLY(bwd_w_f4_t<mx4_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf == 3 && gybf == 3) bwd_w_f4_t<mx8_t, mx8_t>(x, xs, (const mx8_t *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3 && gybf == 2) bwd_w_f4_t<mx8_t, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 3 && gybf == 1) VERIFY_ONLY(bwd_w_f4_t<mx8_t, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf == 3) VERIFY_ONLY(bwd_w_f4_t<mx8_t, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf == 2 && gybf) bwd_w_f4_t<__half, __half>(x, xs, (const __half *)gy, ys, gw, gb, gp, sp, had);
    else if (xbf == 2) VERIFY_ONLY(bwd_w_f4_t<__half, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf && gybf) VERIFY_ONLY(bwd_w_f4_t<bf16, bf16>(x, xs, (const bf16 *)gy, ys, gw, gb, gp, sp, had));
    else if (xbf) VERIFY_ONLY(bwd_w_f4_t<bf16, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had));
    else if (gybf) { fprintf(stderr, "lp_bwd_w_f4: bf16 gradient with fp32 activations is not supported\n"); abort(); }
    else VERIFY_ONLY(bwd_w_f4_t<float, float>(x, xs, (const float *)gy, ys, gw, gb, gp, sp, had));
    LPCK();
    return 0;
}
/* test probe: mean of n direct-nibble stochastic roundings of v (grid units, |v| <= 6) */
__global__ void sr_nib_probe_k(float v, size_t n, double *acc) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(acc, (double)dec_e2m1n(sr_e2m1_nib(v, sr_hash(0x7654321u, i) & 0xffffu)));
}
extern "C" double lp_sr_e2m1_nib_mean(float v, size_t n) {
    double *d; cudaMalloc(&d, sizeof(double)); cudaMemset(d, 0, sizeof(double));
    sr_nib_probe_k<<<nblk_(n, 256), 256>>>(v, n, d);
    double h = 0; cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d); LPCK();
    return h / (double)n;
}
