/* explicit instantiation of the tensor-core forward dispatch (kernels compiled in parallel units) */
#include "nn_common.cuh"
#ifdef UFSM_ALL_TYPES   /* verification type: make VERIFY=1 */
template int conv_fwd_tc_f16acc<bf16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
#endif
