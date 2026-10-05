/* FP8 (e4m3, MX) 3^3 stride-1 forward kernels; owns the shared error state. */
#include "lp_f8fwd.cuh"

cudaError_t g_lp_err = cudaSuccess;
/* instantiated in lp_f8fwd_i*.cu */
extern template void fwd_f8_t<mx4_t, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f8_t<mx8_t, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f8_t<__half, __half>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void fwd_f8_t<bf16, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void fwd_f8_t<float, float>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
extern template void fwd_f8_t<mx8_t, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f8_t<mx4_t, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f8_t<__half, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f8_t<__half, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void fwd_f8_t<bf16, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void fwd_f8_t<bf16, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif

extern "C" int lp_conv_fwd_f8(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    if ((xbf == 3 || xbf == 4) && (ybf == 3 || ybf == 4) && xbf != ybf) {
        if (xs.c > 8 || sp.x2 || sp.y2 || sp.up || sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: mixed MX input/output only for the network stem (Ci <= 8)\n"); abort(); }
        if (xbf == 3) fwd_f8_t<mx8_t, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else fwd_f8_t<mx4_t, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        LPCK();
        return 0;
    }
    if ((xbf == 1 || xbf == 2) && ybf >= 3) {   /* 16-bit network input -> MX activation (tap-packed small kernel) */
        if (xs.c > 8 || sp.x2 || sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: 16-bit in / MX out only for the network input (Ci <= 8)\n"); abort(); }
        if (xbf == 2 && ybf == 4) fwd_f8_t<__half, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 2) fwd_f8_t<__half, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (ybf == 4) VERIFY_ONLY(fwd_f8_t<bf16, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp));
        else VERIFY_ONLY(fwd_f8_t<bf16, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp));
        LPCK();
        return 0;
    }
    int dt = lp_dtype_check("lp_conv_fwd_f8", xbf, ybf);
    if (dt >= 3 && sp.accum) { fprintf(stderr, "lp_conv_fwd_f8: MX storage with accumulate is not supported\n"); abort(); }
    if (dt == 4) fwd_f8_t<mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt == 3) fwd_f8_t<mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt == 2) fwd_f8_t<__half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (dt) VERIFY_ONLY(fwd_f8_t<bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp));
    else VERIFY_ONLY(fwd_f8_t<float>(x, xs, w, b, cout, y, gp, osum, Go, sp));
    LPCK();
    return 0;
}
/* weight prep kernels shared by the units of this family (declared in lp_common.cuh) */
__global__ void prep_w8_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP) {
    const int nch = Cip / 32;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)27 * Cop * nch) return;
    int ch = (int)(i % nch), cop = (int)((i / nch) % Cop), t = (int)(i / ((size_t)nch * Cop));
    int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);   /* padded output row -> real output channel (-1 = padding) */
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
    int e = mx_exp(amax, 1.f / 448.f);
    float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + ((size_t)t * Cop + cop) * Cip + ch * 32);   /* padded row (co is the real channel or -1) */
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}
__global__ void prep_w8p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)P16_KS * Cop) return;
    const int cop = (int)(i % Cop), ks = (int)(i / Cop), r = ks >> 1, h = ks & 1;
    const int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const int kx = 2 * h + (k >> 4), ci = k & 15;
        v[k] = (kx < 3 && co >= 0 && co < Co && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + r * 3 + kx] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const int e = mx_exp(amax, 1.f / 448.f);
    const float m = exp2i(-e);
    uint4 *dst = (uint4 *)(wq + i * 32);
    dst[0] = make_uint4(cvt_e4m3x4(v[0] * m, v[1] * m, v[2] * m, v[3] * m), cvt_e4m3x4(v[4] * m, v[5] * m, v[6] * m, v[7] * m),
                        cvt_e4m3x4(v[8] * m, v[9] * m, v[10] * m, v[11] * m), cvt_e4m3x4(v[12] * m, v[13] * m, v[14] * m, v[15] * m));
    dst[1] = make_uint4(cvt_e4m3x4(v[16] * m, v[17] * m, v[18] * m, v[19] * m), cvt_e4m3x4(v[20] * m, v[21] * m, v[22] * m, v[23] * m),
                        cvt_e4m3x4(v[24] * m, v[25] * m, v[26] * m, v[27] * m), cvt_e4m3x4(v[28] * m, v[29] * m, v[30] * m, v[31] * m));
    ws[i] = (uint8_t)(e + 127);
}
