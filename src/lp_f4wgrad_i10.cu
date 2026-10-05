/* explicit instantiations of bwd_w_f4_t */
#include "lp_f4wgrad.cuh"
#ifdef UFSM_ALL_TYPES   /* verification type: make VERIFY=1 */
template void bwd_w_f4_t<bf16, bf16>(const void *x, shape5 xs, const bf16 *gy, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp, int had);
#endif
