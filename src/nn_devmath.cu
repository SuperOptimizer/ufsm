// the device C library's trigonometry by pi (tests/test_devmath.c): sincospi / sinpi / cospi / sincos in double and sincospif in float,
// one argument per thread. Until mcc e0e23d0 (11-27) the device library had no double sinpi / cospi: mcuda_device.h's sincospi called
// them, mcudahw refused the launch ("device code calls sinpi, which nothing defines") and nn_aug.cu's FFT twiddle table (four_tab_k)
// stayed unwritten (inkaug).
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void devmath_k(const double *x, double *o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s, c, s2, c2; float fs, fc;
    sincospi(x[i], &s, &c);
    sincos(x[i], &s2, &c2);
    sincospif((float)x[i], &fs, &fc);
    double *r = o + (size_t)i * 8;
    r[0] = s; r[1] = c; r[2] = sinpi(x[i]); r[3] = cospi(x[i]); r[4] = s2; r[5] = c2; r[6] = fs; r[7] = fc;
}

/* out[8 n]: per argument sincospi (s, c), sinpi, cospi, sincos (s, c), sincospif (s, c); 0 or the CUDA error */
extern "C" int nn_devmath_probe(const double *x, int n, double *out) {
    double *dx = nullptr, *d = nullptr;
    cudaError_t e = cudaMalloc(&dx, (size_t)n * 8);
    if (!e) e = cudaMalloc(&d, (size_t)n * 64);
    if (!e) e = cudaMemcpy(dx, x, (size_t)n * 8, cudaMemcpyHostToDevice);
    if (!e) { devmath_k<<<(n + 127) / 128, 128>>>(dx, d, n); e = cudaGetLastError(); }
    if (!e) e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(out, d, (size_t)n * 64, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(d);
    return (int)e;
}
