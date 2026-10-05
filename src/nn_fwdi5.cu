/* explicit instantiation of the tensor-core forward dispatch (kernels compiled in parallel units) */
#include "nn_common.cuh"
template int conv_fwd_tc_s2_h<bf16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp);
