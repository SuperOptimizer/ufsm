/* explicit instantiations of bwd_w_f4_t */
#include "lp_f4wgrad.cuh"
template void bwd_w_f4_t<mx4_t, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
template void bwd_w_f4_t<mx4_t, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
