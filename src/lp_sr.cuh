#pragma once
/* exact stochastic rounding onto the e4m3 grid, shared by the fp8 kernels (lp_f8wgrad, lp_f8fwd) */
#include "lp_common.cuh"

/* exact stochastic rounding of a gradient operand onto the e4m3 grid (|v| <= 448: block-scaled), keyed by the element. A value on
   the grid comes back as it is; sr_e4m3's dither then round-to-nearest pulled exact powers of two down (E = v (1 - 1/64): the binade
   below is twice as fine), which MX-fp4 gradients (half of their magnitudes are powers of two) hit constantly. Off the grid the
   two bracketing points are chosen with probability proportional to the distance to the other one: inside a binade the grid is
   uniform, so the floor of the value in grid units plus a uniform. The hash is computed only off the grid. */
__device__ __forceinline__ float sr_e4m3x(float v, uint32_t seed, uint64_t key) {
    const unsigned b = __float_as_uint(v) & 0x7fffffffu;
    const int ex = max((int)(b >> 23) - 127, -6);
    const float t = __uint_as_float(b) * exp2i(3 - ex);   /* grid units: [8, 16) in a normal binade, [0, 8) below 2^-6 */
    if (t == floorf(t)) return v;   /* on the grid (and 0, inf); NaN goes on and stays NaN */
    const float u = (float)(sr_hash(seed, key) >> 8) * (1.f / 16777216.f);
    return copysignf(floorf(t + u) * exp2i(ex - 3), v);
}
