/* explicit instantiations of bwd_w_f8_t */
#include "lp_f8wgrad.cuh"
#ifdef UFSM_ALL_TYPES   /* verification type: make VERIFY=1 */
template void bwd_w_f8_t<float, float>(const void *x, shape5 xs, const float *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp);
#endif
