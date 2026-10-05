/* explicit instantiations of fwd_f8_t */
#include "lp_f8fwd.cuh"
template void fwd_f8_t<bf16, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
template void fwd_f8_t<float, float>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
