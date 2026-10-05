/* explicit instantiations of bwd_w_f8_t */
#include "lp_f8wgrad.cuh"
template void bwd_w_f8_t<mx8_t, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
