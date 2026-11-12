/* the normal-pooled MIL ink loss on the GPU (src/inkmil.h): one thread per ray, its samples in registers, the gradient by atomics. */
#include <cuda_runtime.h>
#include <math.h>
#include "inkmil.h"

enum { KMAX = 256 };
__device__ static inline float tri_f(const float *v, int D, int H, int W, float z, float y, float x, int c[3], float f[3]) {   /* (D, H, W >= 2) */
    if (!(z >= 0 && y >= 0 && x >= 0 && z <= D - 1 && y <= H - 1 && x <= W - 1)) return NAN;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x; if (z0 > D - 2) z0 = D - 2; if (y0 > H - 2) y0 = H - 2; if (x0 > W - 2) x0 = W - 2;
    const float fz = z - z0, fy = y - y0, fx = x - x0; c[0] = z0; c[1] = y0; c[2] = x0; f[0] = fz; f[1] = fy; f[2] = fx;
    const size_t b = ((size_t)z0 * H + y0) * W + x0, sy = W, sz = (size_t)H * W;
    const float a00 = v[b] * (1 - fx) + v[b + 1] * fx, a01 = v[b + sy] * (1 - fx) + v[b + sy + 1] * fx;
    const float a10 = v[b + sz] * (1 - fx) + v[b + sz + 1] * fx, a11 = v[b + sz + sy] * (1 - fx) + v[b + sz + sy + 1] * fx;
    return (a00 * (1 - fy) + a01 * fy) * (1 - fz) + (a10 * (1 - fy) + a11 * fy) * fz;
}
__global__ void k_mil(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, int K, float tau, float eps, float scale, double *loss) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nr) return;
    const inkray r = rays[i]; float z[KMAX]; int nv = 0, am = -1; float zm = -INFINITY;
    for (int k = 0; k < K; k++) {
        const float t = -R + k * step; int c[3]; float f[3];
        const float v = tri_f(logit, D, H, W, r.p[0] + t * r.n[0], r.p[1] + t * r.n[1], r.p[2] + t * r.n[2], c, f);
        z[k] = v; if (v == v) { nv++; if (v > zm) { zm = v; am = k; } }
    }
    if (!nv) return;
    float zr, se = 0;
    if (tau > 0) { for (int k = 0; k < K; k++) if (z[k] == z[k]) se += expf((z[k] - zm) / tau); zr = zm + tau * logf(se / nv); } else zr = zm;
    const float y = r.y ? 1 - eps / 2 : eps / 2;
    const float l = fmaxf(zr, 0) - zr * y + log1pf(expf(-fabsf(zr)));
    atomicAdd(loss, (double)(scale * r.w * l));
    const float g = scale * r.w * (1 / (1 + expf(-zr)) - y);
    for (int k = 0; k < K; k++) {
        if (!(z[k] == z[k])) continue;
        const float dk = tau > 0 ? expf((z[k] - zm) / tau) / se : (k == am ? 1.f : 0.f); if (dk == 0) continue;
        const float t = -R + k * step; int c[3]; float f[3];
        tri_f(logit, D, H, W, r.p[0] + t * r.n[0], r.p[1] + t * r.n[1], r.p[2] + t * r.n[2], c, f);
        for (int q = 0; q < 8; q++) {
            const int a = c[0] + (q >> 2 & 1), b = c[1] + (q >> 1 & 1), e = c[2] + (q & 1);
            const float wq = ((q >> 2 & 1) ? f[0] : 1 - f[0]) * ((q >> 1 & 1) ? f[1] : 1 - f[1]) * ((q & 1) ? f[2] : 1 - f[2]);
            if (wq != 0) atomicAdd(&grad[((size_t)a * H + b) * W + e], g * dk * wq);
        }
    }
}
extern "C" int ink_mil_loss_dev(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale, double *loss, void *stream) {
    if (nr <= 0) return 0;
    const int K = (int)floorf(2 * R / step + 1e-4f) + 1; if (K > KMAX || K < 1) return -2;
    k_mil<<<(nr + 127) / 128, 128, 0, (cudaStream_t)stream>>>(logit, grad, D, H, W, rays, nr, R, step, K, tau, eps, scale, loss);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
/* tests: the same on host memory (copies in and out; grad accumulated into the host array) */
extern "C" int ink_mil_loss_hostmem(const float *logit, float *grad, int D, int H, int W, const inkray *rays, int nr, float R, float step, float tau, float eps, float scale, double *loss) {
    const size_t nv = (size_t)D * H * W; float *dl = nullptr, *dg = nullptr; inkray *dr = nullptr; double *dL = nullptr; int rc = -1;
    if (cudaMalloc(&dl, nv * 4) || cudaMalloc(&dg, nv * 4) || cudaMalloc(&dr, (size_t)(nr ? nr : 1) * sizeof *dr) || cudaMalloc(&dL, 8)) goto out;
    if (cudaMemcpy(dl, logit, nv * 4, cudaMemcpyHostToDevice) || cudaMemcpy(dg, grad, nv * 4, cudaMemcpyHostToDevice) || cudaMemcpy(dr, rays, (size_t)nr * sizeof *dr, cudaMemcpyHostToDevice) ||
        cudaMemcpy(dL, loss, 8, cudaMemcpyHostToDevice)) goto out;
    if (ink_mil_loss_dev(dl, dg, D, H, W, dr, nr, R, step, tau, eps, scale, dL, nullptr)) goto out;
    if (cudaMemcpy(grad, dg, nv * 4, cudaMemcpyDeviceToHost) || cudaMemcpy(loss, dL, 8, cudaMemcpyDeviceToHost)) goto out;
    rc = 0;
out:
    cudaFree(dl); cudaFree(dg); cudaFree(dr); cudaFree(dL); return rc;
}
