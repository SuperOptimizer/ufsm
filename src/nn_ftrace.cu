/* ufsm fiber-trace on the GPU (src/ftgpu.h): the per-voxel fibre axis of src/ftaxis.c ft_ridge_axis, which is ~97 % of a --mask trace
   on the CPU (512 x 576 x 768 box: 128 s, the streamline integration ~3 s). Byte-identical to the CPU: the gradient products and their
   (2R+1)^3 sums are exact integers in both (R <= 6: every partial float sum of the CPU stays below 2^24), and the eigenvector power
   iterations run the CPU's double operations in its order; this object is built with -fmad=false and without fast math (Makefile), so
   division / sqrt are IEEE and nothing is contracted (gcc -std=c23 does not contract either).
   One thread per (y, x) column marches in z: the in-plane (2R+1)^2 sums of the six products of plane k are formed once per plane and kept
   in a ring of 2R+1 planes; the z sum slides (T(z+1) = T(z) + S(z+1+R) - S(z-R), planes clamped at the ends as the CPU's ring). */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "ftgpu.h"

#define FT_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "fiber-trace gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); goto fail; } } while (0)
enum { FT_RMAX = 6 };

__device__ __forceinline__ int64_t clampi(int64_t v, int64_t n) { return v < 0 ? 0 : v >= n ? n - 1 : v; }

/* the six products of plane k summed over the (2R+1)^2 in-plane box at (y, x), every coordinate clamped like the CPU's PRV / box sums */
__device__ void plane_sums(const uint8_t *__restrict__ pr, int64_t D, int64_t H, int64_t W, int64_t k, int64_t y, int64_t x, int R, int S[6]) {
    for (int c = 0; c < 6; c++) S[c] = 0;
    const int64_t z = clampi(k, D), zm = clampi(z - 1, D), zp = clampi(z + 1, D);
    for (int64_t dy = -R; dy <= R; dy++) {
        const int64_t yy = clampi(y + dy, H), ym = clampi(yy - 1, H), yp = clampi(yy + 1, H);
        for (int64_t dx = -R; dx <= R; dx++) {
            const int64_t xx = clampi(x + dx, W), xm = clampi(xx - 1, W), xp = clampi(xx + 1, W);
            const int gz = (int)pr[(zp * H + yy) * W + xx] - (int)pr[(zm * H + yy) * W + xx];
            const int gy = (int)pr[(z * H + yp) * W + xx] - (int)pr[(z * H + ym) * W + xx];
            const int gx = (int)pr[(z * H + yy) * W + xp] - (int)pr[(z * H + yy) * W + xm];
            S[0] += gz * gz; S[1] += gy * gy; S[2] += gx * gx; S[3] += gz * gy; S[4] += gz * gx; S[5] += gy * gx;
        }
    }
}

/* the CPU's eigen step (src/ftaxis.c), operation for operation */
__device__ void axis_of(const int Ti[6], char prior, uint8_t *onx, uint8_t *ony) {
    const double T[6] = {(double)Ti[0], (double)Ti[1], (double)Ti[2], (double)Ti[3], (double)Ti[4], (double)Ti[5]};
    *onx = *ony = 128; const double tr = T[0] + T[1] + T[2]; if (!(tr > 0)) return;
    const double M[6] = {tr - T[0], tr - T[1], tr - T[2], -T[3], -T[4], -T[5]}; double v[3] = {0.577, 0.577, 0.577};
    for (int it = 0; it < 40; it++) {
        const double w[3] = {M[0] * v[0] + M[3] * v[1] + M[4] * v[2], M[3] * v[0] + M[1] * v[1] + M[5] * v[2], M[4] * v[0] + M[5] * v[1] + M[2] * v[2]};
        const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; v[0] = w[0] / m; v[1] = w[1] / m; v[2] = w[2] / m;
    }
    if (prior) {
        double u[3] = {0.577, 0.577, 0.577};
        for (int it = 0; it < 40; it++) {
            const double w[3] = {T[0] * u[0] + T[3] * u[1] + T[4] * u[2], T[3] * u[0] + T[1] * u[1] + T[5] * u[2], T[4] * u[0] + T[5] * u[1] + T[2] * u[2]};
            const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; u[0] = w[0] / m; u[1] = w[1] / m; u[2] = w[2] / m;
        }
        double a[3];
        if (prior == 'H') { a[0] = 0; a[1] = u[2]; a[2] = -u[1]; }
        else { a[0] = 1 - u[0] * u[0]; a[1] = -u[0] * u[1]; a[2] = -u[0] * u[2]; }
        const double m = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (m > 1e-6) for (int c = 0; c < 3; c++) v[c] = a[c] / m;
    }
    if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; }
    *onx = (uint8_t)lround(128 + 127 * v[2]); *ony = (uint8_t)lround(128 + 127 * v[1]);
}

__global__ void k_ridge_axis(const uint8_t *__restrict__ pr, int64_t D, int64_t H, int64_t W, int R, char prior, int64_t z0, int64_t z1,
                             uint8_t *__restrict__ nx, uint8_t *__restrict__ ny) {
    const int64_t x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, y = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int RW = 2 * R + 1; int ring[2 * FT_RMAX + 1][6], T[6] = {0, 0, 0, 0, 0, 0};
    for (int64_t k = z0 - R; k <= z0 + R; k++) { int *S = ring[(k - (z0 - R)) % RW]; plane_sums(pr, D, H, W, k, y, x, R, S); for (int c = 0; c < 6; c++) T[c] += S[c]; }
    for (int64_t z = z0; z < z1; z++) {
        if (z > z0) {   /* slide: drop plane z-1-R (ring slot of k = z-1-R), add plane z+R into the same slot */
            const int64_t kout = z - 1 - R, kin = z + R; int *S = ring[(kout - (z0 - R)) % RW];
            for (int c = 0; c < 6; c++) T[c] -= S[c];
            plane_sums(pr, D, H, W, kin, y, x, R, S); for (int c = 0; c < 6; c++) T[c] += S[c];
        }
        const size_t j = ((size_t)z * H + y) * W + x; axis_of(T, prior, nx + j, ny + j);
    }
}

extern "C" int ft_ridge_axis_gpu(int gpu, const uint8_t *pr, const int64_t n[3], uint8_t *nx, uint8_t *ny, int R, char prior) {
    if (R < 1 || R > FT_RMAX) { fprintf(stderr, "fiber-trace gpu: --st-r %d (the GPU path takes 1..%d)\n", R, FT_RMAX); return 1; }
    const size_t nv = (size_t)n[0] * n[1] * n[2]; uint8_t *dp = nullptr, *dx = nullptr, *dy = nullptr; int rc = 1;
    FT_CK(cudaSetDevice(gpu));
    FT_CK(cudaMalloc(&dp, nv)); FT_CK(cudaMalloc(&dx, nv)); FT_CK(cudaMalloc(&dy, nv));
    FT_CK(cudaMemcpy(dp, pr, nv, cudaMemcpyHostToDevice));
    {   /* z chunks of 32 planes per launch: many short columns keep every SM busy (442 k columns x 16 chunks for a 512 x 576 x 768 box) */
        const dim3 bs(32, 4), gs((unsigned)((n[2] + 31) / 32), (unsigned)((n[1] + 3) / 4));
        for (int64_t z0 = 0; z0 < n[0]; z0 += 32) {
            const int64_t z1 = z0 + 32 < n[0] ? z0 + 32 : n[0];
            k_ridge_axis<<<gs, bs>>>(dp, n[0], n[1], n[2], R, prior, z0, z1, dx, dy); FT_CK(cudaGetLastError());
        }
    }
    FT_CK(cudaMemcpy(nx, dx, nv, cudaMemcpyDeviceToHost)); FT_CK(cudaMemcpy(ny, dy, nv, cudaMemcpyDeviceToHost));
    rc = 0;
fail:
    cudaFree(dp); cudaFree(dx); cudaFree(dy); if (rc) cudaGetLastError();
    return rc;
}
