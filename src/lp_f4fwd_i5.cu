/* explicit instantiations of fwd_f4_t */
#include "lp_f4fwd.cuh"
template void fwd_f4_t<mx8_t, __half>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
template void fwd_f4_t<mx8_t, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
