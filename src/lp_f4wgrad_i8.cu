/* explicit instantiations of bwd_w_f4_t */
#include "lp_f4wgrad.cuh"
template void bwd_w_f4_t<__half, __half>(const void *x, shape5 xs, const __half *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
