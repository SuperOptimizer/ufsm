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
