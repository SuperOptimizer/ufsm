/* Force-included into every CUDA unit compiled by clang (-include, CMakeLists.txt): what nvcc's CUDA headers provide that clang's
   CUDA wrappers do not.
   - Host overloads of min / max. nvcc's crt/math_functions.hpp defines them __host__ __device__ for every integer width, the
     mixed signed / unsigned pairs and float / double, so host code in the .cu files (launch geometry, smem sizes) calls
     min(a, b) on ints; clang declares only the __device__ ones. These are __host__ overloads with nvcc's signatures and
     semantics (mixed signed / unsigned compares as unsigned, float / double mixes as double); clang's target-based overloading
     picks them in host code and in the host pass of __host__ __device__ functions, and its own in device code. */
#pragma once
#if defined(__clang__) && defined(__CUDA__)
#define UFSM_HMM(T, A, B)                                                                                                         \
    static inline __host__ T min(A a, B b) { return (T)a < (T)b ? (T)a : (T)b; }                                               \
    static inline __host__ T max(A a, B b) { return (T)a > (T)b ? (T)a : (T)b; }
UFSM_HMM(int, int, int)
UFSM_HMM(unsigned int, unsigned int, unsigned int)
UFSM_HMM(unsigned int, int, unsigned int)
UFSM_HMM(unsigned int, unsigned int, int)
UFSM_HMM(long, long, long)
UFSM_HMM(unsigned long, unsigned long, unsigned long)
UFSM_HMM(unsigned long, long, unsigned long)
UFSM_HMM(unsigned long, unsigned long, long)
UFSM_HMM(long long, long long, long long)
UFSM_HMM(unsigned long long, unsigned long long, unsigned long long)
UFSM_HMM(unsigned long long, long long, unsigned long long)
UFSM_HMM(unsigned long long, unsigned long long, long long)
#undef UFSM_HMM
/* float / double as fminf / fmin (nvcc: NaN-ignoring, like fmin) */
static inline __host__ float min(float a, float b) { return __builtin_fminf(a, b); }
static inline __host__ float max(float a, float b) { return __builtin_fmaxf(a, b); }
static inline __host__ double min(double a, double b) { return __builtin_fmin(a, b); }
static inline __host__ double max(double a, double b) { return __builtin_fmax(a, b); }
static inline __host__ double min(float a, double b) { return __builtin_fmin(a, b); }
static inline __host__ double max(float a, double b) { return __builtin_fmax(a, b); }
static inline __host__ double min(double a, float b) { return __builtin_fmin(a, b); }
static inline __host__ double max(double a, float b) { return __builtin_fmax(a, b); }
#endif
/* - nvcc -use_fast_math also maps expf / exp10f / log10f / tanf / powf to their intrinsics in device code; clang's
     -fgpu-approx-transcendentals maps only sin / cos / sincos / log / log2 (__clang_cuda_math.h __FAST_OR_SLOW). Without this the
     fast units computed expf precisely: different bits from nvcc, and two kernels that must agree bit for bit (nn_loss_pack vs
     nn_loss_async_tol: __expf in one, expf in the other; tests/test_half.c) disagreed. Device pass of the fast units only
     (UFSM_DEVICE_FAST_MATH, CMakeLists.txt UFSM_CU_FAST); host code in the same file keeps the libm functions. */
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__) && defined(UFSM_DEVICE_FAST_MATH)
#define expf(x) __expf(x)
#define exp10f(x) __exp10f(x)
#define log10f(x) __log10f(x)
#define tanf(x) __tanf(x)
#define powf(x, y) __powf(x, y)
#endif
