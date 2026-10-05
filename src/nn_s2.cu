/* CUDA ops: s2 section of the former nn.cu */
#include "nn_common.cuh"

int cur_dev(void) { int d = 0; cudaGetDevice(&d); return d & 7; }
