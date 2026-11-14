/* the head-gradient plane scale chosen on the device (hgs.h; 2026-11-14). A UFSM_STRICT_UNITS unit: the bound's divisions and products are
   IEEE and uncontracted as in the host C, so the device choice is the host function's bits (test_hgsdev) */
#include "nn_common.cuh"
#include "hgs.h"

__global__ void hgs_choose_k(hgs_args a, float *out) {
    if (threadIdx.x || blockIdx.x) return;
    const int k = hgs_exp(hgs_bound_all(&a));
    out[0] = ldexpf(1.f, k); out[1] = ldexpf(1.f, -k);
}
/* out: device {s, 1 / s}; the terms' pointers are device loss scratches, read after the statistics pass's fins on the same stream */
extern "C" void nn_hgs_choose(const hgs_args *a, float *out) { hgs_choose_k<<<1, 32>>>(*a, out); KCHECK(); }
/* (the host reference is hgs.h's inline functions in host C, built -ffp-contract=off: this unit's host side may contract) */
