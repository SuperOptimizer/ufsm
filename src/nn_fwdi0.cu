/* explicit instantiation of the tensor-core forward dispatch (kernels compiled in parallel units) */
#include "nn_common.cuh"
template int conv_fwd_tc_h<f16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
