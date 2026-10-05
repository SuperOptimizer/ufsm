/* FP4 (e2m1, MX) 3^3 stride-1 forward kernels and the weight memo. */
#include "lp_f4fwd.cuh"

extern "C" void lp_set_w4_2d(int on) { g_w2d = on; lp_wmemo_clear(); }   /* tests; default from UFSM_W4_2D */
extern "C" void lp_wmemo_step(unsigned step) { g_wmemo_step = step; }
extern "C" void lp_wmemo_clear(void) { g_wmemo_gen++; }
int g_w2d = -1;
unsigned g_wmemo_step = 0, g_wmemo_gen = 0;
wmemo_s g_wmemo[8][WMEMO_N];
/* instantiated in lp_f4fwd_i*.cu */
extern template void fwd_f4_t<mx4_t, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f4_t<mx8_t, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
extern template void fwd_f4_t<__half, __half>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void fwd_f4_t<bf16, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void fwd_f4_t<float, float>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
#ifdef UFSM_ALL_TYPES
extern template void fwd_f4_t<mx4_t, float>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
extern template void fwd_f4_t<mx8_t, __half>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#ifdef UFSM_ALL_TYPES
extern template void fwd_f4_t<mx8_t, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif

extern "C" int lp_conv_fwd_f4(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp) {
    if (xs.c <= 8) return lp_conv_fwd_f8(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp);   /* the network input (enc0.c1): fp8 tap-packed kernel */
    if (sp.accum && (xbf >= 3 || ybf >= 3)) { fprintf(stderr, "lp_conv_fwd_f4: MX storage with accumulate is not supported\n"); abort(); }
    if (xbf == ybf) {
        if (xbf == 4) fwd_f4_t<mx4_t, mx4_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 3) fwd_f4_t<mx8_t, mx8_t>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf == 2) fwd_f4_t<__half, __half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
        else if (xbf) VERIFY_ONLY(fwd_f4_t<bf16, bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp));
        else VERIFY_ONLY(fwd_f4_t<float, float>(x, xs, w, b, cout, y, gp, osum, Go, sp));
    } else if (xbf == 4 && ybf == 0) VERIFY_ONLY(fwd_f4_t<mx4_t, float>(x, xs, w, b, cout, y, gp, osum, Go, sp));   /* tests: exact staging check with an unquantised output */
    else if (xbf == 3 && ybf == 2) fwd_f4_t<mx8_t, __half>(x, xs, w, b, cout, y, gp, osum, Go, sp);
    else if (xbf == 3 && ybf == 1) VERIFY_ONLY(fwd_f4_t<mx8_t, bf16>(x, xs, w, b, cout, y, gp, osum, Go, sp));
    else { fprintf(stderr, "lp_conv_fwd_f4: activation types (x %d, y %d) are not instantiated\n", xbf, ybf); abort(); }
    LPCK();
    return 0;
}
/* weight prep kernels shared by the units of this family (declared in lp_f4fwd.cuh) */
__global__ void prep_w4_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP) {
    const int nch = Cip / 32;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)28 * Cop * nch) return;              /* tap 27 = the zero half of the last pair */
    int ch = (int)(i % nch), cop = (int)((i / nch) % Cop), t = (int)(i / ((size_t)nch * Cop));
    int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);   /* padded output row -> real output channel (-1 = padding) */
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (t < 27 && co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
    int e = mx_exp(amax, 1.f / 6.f);
    float m = exp2i(-e);
    *(uint4 *)(wq + ((size_t)t * Cop + cop) * (Cip / 2) + ch * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[i] = (uint8_t)(e + 127);
}
__global__ void prep_w4_2d_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Cip, int Cx, int CxP, int Ox, int OxP) {
    const int nch = Cip / 32, nrt = (Cop + 31) / 32, lane = threadIdx.x & 31;
    const size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    if (tile >= (size_t)28 * nrt * nch) return;
    const int ch = (int)(tile % nch), rt = (int)((tile / nch) % nrt), t = (int)(tile / ((size_t)nch * nrt));
    const int cop = rt * 32 + lane;
    const int co = cop >= Cop ? -1 : Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) { int ci = Cx < 0 ? ch * 32 + k : seg_ci(ch * 32 + k, Ci, Cx, CxP); v[k] = (t < 27 && co >= 0 && co < Co && ci >= 0 && ci < Ci) ? w[((size_t)co * Ci + ci) * 27 + t] : 0.f; amax = fmaxf(amax, fabsf(v[k])); }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const int e = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-e);
    if (cop >= Cop) return;
    *(uint4 *)(wq + ((size_t)t * Cop + cop) * (Cip / 2) + ch * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[((size_t)t * Cop + cop) * nch + ch] = (uint8_t)(e + 127);
}
__global__ void prep_w4p_k(const float *__restrict__ w, uint8_t *__restrict__ wq, uint8_t *__restrict__ ws, int Co, int Ci, int Cop, int Ox, int OxP, int Cs, int cofs) {   /* Cs: w channel stride (-1: Ci), cofs: first channel */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   /* (r, co, block) */
    if (i >= (size_t)9 * Cop * 2) return;
    const int blk = (int)(i & 1), cop = (int)((i >> 1) % Cop), r = (int)((i >> 1) / Cop);
    const int co = Ox < 0 ? cop : seg_ci(cop, Co, Ox, OxP);
    float v[32], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const int kx = 2 * blk + (k >> 4), ci = k & 15;
        v[k] = (kx < 3 && co >= 0 && co < Co && ci < Ci) ? w[((size_t)co * (Cs < 0 ? Ci : Cs) + cofs + ci) * 27 + r * 3 + kx] : 0.f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const int e = mx_exp(amax, 1.f / 6.f);
    const float m = exp2i(-e);
    *(uint4 *)(wq + ((size_t)r * Cop + cop) * 32 + blk * 16) = make_uint4(cvt_e2m1x8(v, m), cvt_e2m1x8(v + 8, m), cvt_e2m1x8(v + 16, m), cvt_e2m1x8(v + 24, m));
    ws[i] = (uint8_t)(e + 127);
}
