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
