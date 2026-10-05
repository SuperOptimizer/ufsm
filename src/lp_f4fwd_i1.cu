/* explicit instantiations of fwd_f4_t */
#include "lp_f4fwd.cuh"
/* tile variants instantiated in the lp_f4fwd_i1<tag>.cu units (tools/gen_lp_launch.py) */
extern template void launch_f4<1, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<1, 4, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<2, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<2, 4, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4<4, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, int Cip, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<1, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<1, 4, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<2, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<2, 4, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<3, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<3, 4, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f4p<4, 2, mx8_t, mx8_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
template void fwd_f4_t<mx8_t, mx8_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
