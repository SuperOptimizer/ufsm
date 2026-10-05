/* explicit instantiations of fwd_f8_t */
#include "lp_f8fwd.cuh"
template void fwd_f8_t<__half, __half>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
