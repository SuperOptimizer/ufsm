/* explicit instantiations of fwd_f4_t */
#include "lp_f4fwd.cuh"
/* tile variants instantiated in the lp_f4fwd_i3<tag>.cu units (tools/gen_lp_launch.py) */
extern template void launch_f4<1, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<1, 4, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<2, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<2, 4, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<4, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<1, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<1, 4, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<2, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<2, 4, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<3, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<3, 4, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<4, 2, bf16, bf16>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
#ifdef UFSM_ALL_TYPES   /* verification type: make VERIFY=1 */
template void fwd_f4_t<bf16, bf16>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
#endif
