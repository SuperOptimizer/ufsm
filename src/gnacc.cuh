#pragma once
/* GroupNorm (sum, sum of squares) accumulators as exact fixed point, so the statistics do not depend on the order of the blocks'
   atomic adds (double atomics of float block partials rounded differently run to run: legacy and mx8 nets gave different GN
   statistics, hence different outputs, on identical inputs). A sum is a slot of GN_SW 64-bit words: lo (the fraction, units of
   2^-64), hi (the signed integer part, two's complement with lo) and a count of non-finite contributions. Adds are integer atomics
   (exact, modular, order-independent); |sum| < 2^62. The buffers stay double * in every signature: sum i lives at word GN_SW * i
   (gn_slot / gn_acc_*); a slot buffer of n sums takes n * GN_SLOT_BYTES (gn_dsums allocates in slots). */
#define GN_SW 3
#define GN_SLOT_BYTES ((size_t)GN_SW * 8)
static inline double *gn_slot(double *b, size_t i) { return (double *)((unsigned long long *)b + (size_t)GN_SW * i); }
#ifdef __CUDACC__
__device__ __forceinline__ void gn_acc_add(double *b, size_t i, double v) {
    unsigned long long *w = (unsigned long long *)b + (size_t)GN_SW * i;
    if (!(fabs(v) < 4.611686018427387904e18)) { atomicAdd(&w[2], 1ull); return; }   /* NaN, inf or out of range: the sum reads as NaN */
    const double f = floor(v);
    const long long hi = (long long)f;
    const unsigned long long lo = (unsigned long long)((v - f) * 18446744073709551616.0);   /* (v - f) in [0, 1): exact scaling, truncation below 2^-64 */
    if (lo) { const unsigned long long o = atomicAdd(&w[0], lo); if (o + lo < o) atomicAdd(&w[1], 1ull); }   /* the carry out of the fraction */
    if (hi) atomicAdd(&w[1], (unsigned long long)hi);
}
__device__ __forceinline__ double gn_acc_get(const double *b, size_t i) {
    const unsigned long long *w = (const unsigned long long *)b + (size_t)GN_SW * i;
    if (w[2]) return __longlong_as_double(0x7ff8000000000000ll);
    return (double)(long long)w[1] + (double)w[0] * 5.42101086242752217e-20;   /* hi + lo 2^-64 */
}
__device__ __forceinline__ void gn_acc_set(double *b, size_t i, double v) {   /* (non-atomic) the slot holds exactly v's fixed-point value */
    unsigned long long *w = (unsigned long long *)b + (size_t)GN_SW * i;
    w[0] = w[1] = w[2] = 0ull;
    if (!(fabs(v) < 4.611686018427387904e18)) { w[2] = 1ull; return; }
    const double f = floor(v);
    w[1] = (unsigned long long)(long long)f; w[0] = (unsigned long long)((v - f) * 18446744073709551616.0);
}
#endif
