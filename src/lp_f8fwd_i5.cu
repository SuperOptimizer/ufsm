/* explicit instantiations of fwd_f8_t */
#include "lp_f8fwd.cuh"
/* tile variants instantiated in the lp_f8fwd_i5<tag>.cu units (tools/gen_lp_launch.py) */
extern template void launch_f8s<1, 4, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<1, 8, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<1, 16, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<2, 4, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<2, 8, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<2, 16, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<4, 4, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<4, 8, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
extern template void launch_f8s<4, 16, mx8_t, mx4_t>(dim3 grid, const void *x, shape5 xs, const uint8_t *wq, const uint8_t *ws, const float *b, int cout, void *y, int Cop, gnp_t gp, double *osum, int Go, split_t sp);
template void fwd_f8_t<mx8_t, mx4_t>(const void *x, shape5 xs, const float *w, const float *b, int cout, void *y, gnp_t gp, double *osum, int Go, split_t sp);
