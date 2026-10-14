/* CUDA kernels of the winding fitter (src/fit.c, src/fit.h). Per field voxel: U at the centre and at +-h/2 along each axis
   (U = r / pitch(z) + offset(z) + trilinear delta), W = U - dir theta / 2 pi, grad W by central differences with the theta
   difference wrapped (so the branch cut never enters), the data terms, and dL/dU at the seven points scattered to the lattice
   corners and the knots (knots through per-block shared sums). */
#include "nn_common.cuh"
extern "C" {
#include "fit.h"
}

#define FIT_MAXK 512
#define FIT_TPB 256

__device__ __forceinline__ float fit_lerp_tab(const float *v, int n, float z) {   /* linear interpolation of a table at index z (clamped) */
    if (z <= 0.f) return v[0];
    if (z >= (float)(n - 1)) return v[n - 1];
    const int i = (int)z; const float t = z - (float)i;
    return v[i] + t * (v[i + 1] - v[i]);
}
struct fit_pt { float U, th, r, p; int k0; float kt; int c0[3]; float ct[3]; };
__device__ __forceinline__ void fit_eval(const fit_dev &f, float z, float y, float x, fit_pt &o) {
    const float cy = fit_lerp_tab(f.cy, f.naxis, z), cx = fit_lerp_tab(f.cx, f.naxis, z);
    const float dy = y - cy, dx = x - cx;
    o.r = sqrtf(dy * dy + dx * dx); o.th = atan2f(dy, dx);
    float kz = z / (float)f.knot_dz; kz = fminf(fmaxf(kz, 0.f), (float)(f.nk - 1));
    o.k0 = min((int)kz, f.nk - 2 < 0 ? 0 : f.nk - 2); o.kt = f.nk > 1 ? kz - (float)o.k0 : 0.f;
    const int k1 = f.nk > 1 ? o.k0 + 1 : o.k0;
    o.p = f.pitch[o.k0] + o.kt * (f.pitch[k1] - f.pitch[o.k0]);
    const float off = f.offset[o.k0] + o.kt * (f.offset[k1] - f.offset[o.k0]);
    const float q[3] = {z / (float)f.L, y / (float)f.L, x / (float)f.L}; const int dim[3] = {f.lz, f.ly, f.lx};
    for (int d = 0; d < 3; d++) {
        float a = fminf(fmaxf(q[d], 0.f), (float)(dim[d] - 1));
        o.c0[d] = min((int)a, dim[d] - 2 < 0 ? 0 : dim[d] - 2); o.ct[d] = dim[d] > 1 ? a - (float)o.c0[d] : 0.f;
    }
    float del = 0.f;
    for (int c = 0; c < 8; c++) {
        const int a = o.c0[0] + (c >> 2 & 1), b = o.c0[1] + (c >> 1 & 1), e = o.c0[2] + (c & 1);
        const float wt = (c >> 2 & 1 ? o.ct[0] : 1.f - o.ct[0]) * (c >> 1 & 1 ? o.ct[1] : 1.f - o.ct[1]) * (c & 1 ? o.ct[2] : 1.f - o.ct[2]);
        if (wt != 0.f) del += wt * f.delta[((size_t)min(a, f.lz - 1) * f.ly + min(b, f.ly - 1)) * f.lx + min(e, f.lx - 1)];
    }
    o.U = o.r / o.p + off + del;
}
__device__ __forceinline__ float fit_packed(const fit_dev &f, size_t i) {   /* the phase weight factor of packed cells */
    if (f.packed_sp <= 0.f || !f.sp[i]) return 1.f;
    return fminf(1.f, fmaxf(f.packed_floor, ((float)f.sp[i] - 0.6f * f.packed_sp) / (0.4f * f.packed_sp)));
}
__device__ __forceinline__ float fit_wrap(float a) {   /* into (-pi, pi] */
    const float tp = 6.283185307179586f;
    a -= tp * floorf((a + 3.14159265358979f) / tp);
    return a;
}

__global__ void fit_loss_k(fit_dev f, double *parts, float *g_delta, float *g_pitch, float *g_offset) {
    __shared__ float sp_pitch[FIT_MAXK], sp_off[FIT_MAXK];
    __shared__ double red[5][FIT_TPB];
    for (int k = threadIdx.x; k < f.nk; k += blockDim.x) { sp_pitch[k] = 0.f; sp_off[k] = 0.f; }
    __syncthreads();
    double lp[5] = {0, 0, 0, 0, 0};
    const size_t n = (size_t)f.D * f.H * f.W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const float wv = i < n ? f.w[i] * (1.f / 255.f) : 0.f;
    if (wv > 0.f) {
        const int z = (int)(i / ((size_t)f.H * f.W)), y = (int)((i / f.W) % f.H), x = (int)(i % f.W);
        const float e = 0.5f * f.h, pz = (z + 0.5f) * f.h, py = (y + 0.5f) * f.h, px = (x + 0.5f) * f.h;
        fit_pt q[7];
        fit_eval(f, pz, py, px, q[0]);
        if (q[0].r >= f.core * q[0].p) {
            fit_eval(f, pz + e, py, px, q[1]); fit_eval(f, pz - e, py, px, q[2]);
            fit_eval(f, pz, py + e, px, q[3]); fit_eval(f, pz, py - e, px, q[4]);
            fit_eval(f, pz, py, px + e, q[5]); fit_eval(f, pz, py, px - e, q[6]);
            const float i2p = 0.15915494309189535f, sdir = (float)f.dir;
            float g[3];
            for (int d = 0; d < 3; d++) g[d] = ((q[1 + 2 * d].U - q[2 + 2 * d].U) - sdir * fit_wrap(q[1 + 2 * d].th - q[2 + 2 * d].th) * i2p) / (2.f * e);
            const float W0 = q[0].U - sdir * q[0].th * i2p;
            float dU[7] = {0, 0, 0, 0, 0, 0, 0}, dg[3] = {0, 0, 0};
            /* phase */
            const float mc = ((int)f.pc[i] - 128) * (1.f / 127.f), ms = ((int)f.ps[i] - 128) * (1.f / 127.f), conf = sqrtf(mc * mc + ms * ms);
            if (conf > 0.05f && f.w_phase > 0.f) {
                const float ph = atan2f(ms, mc), a = 6.283185307179586f * W0 - ph;
                float sa, ca; sincosf(a, &sa, &ca);
                const float c = f.w_phase * wv * fminf(conf, 1.f) * fit_packed(f, i);
                lp[1] += c * (1.f - ca); dU[0] += c * 6.283185307179586f * sa;
            }
            const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + 1e-12f, gl = sqrtf(g2);
            /* normal: 1 - g^T T g / (|g|^2 tr T) */
            const size_t pl = n;
            const float tzz = f.t[i] * (1.f / 255.f), tyy = f.t[pl + i] * (1.f / 255.f), txx = f.t[2 * pl + i] * (1.f / 255.f);
            const float tzy = f.t[3 * pl + i] * (1.f / 255.f) - 0.5f, tzx = f.t[4 * pl + i] * (1.f / 255.f) - 0.5f, tyx = f.t[5 * pl + i] * (1.f / 255.f) - 0.5f;
            const float tr = tzz + tyy + txx;
            if (tr > 0.1f && f.w_normal > 0.f) {
                const float Tg[3] = {tzz * g[0] + tzy * g[1] + tzx * g[2], tzy * g[0] + tyy * g[1] + tyx * g[2], tzx * g[0] + tyx * g[1] + txx * g[2]};
                const float gTg = g[0] * Tg[0] + g[1] * Tg[1] + g[2] * Tg[2], c = f.w_normal * wv;
                lp[2] += c * (1.f - gTg / (g2 * tr));
                for (int d = 0; d < 3; d++) dg[d] += -c * (2.f * Tg[d] / (g2 * tr) - 2.f * gTg * g[d] / (g2 * g2 * tr));
            }
            /* density and non-degeneracy */
            if (f.sp[i]) {
                const float rho = 8.f / (f.h * (float)f.sp[i]), lr = logf(gl) - logf(rho);
                if (f.w_density > 0.f) { const float c = f.w_density * wv; lp[3] += c * lr * lr; for (int d = 0; d < 3; d++) dg[d] += c * 2.f * lr * g[d] / g2; }
                const float lo = 0.4f * rho - gl;
                if (lo > 0.f && f.w_nondeg > 0.f) { const float c = f.w_nondeg * wv; lp[4] += c * lo * lo; for (int d = 0; d < 3; d++) dg[d] += -c * 2.f * lo * g[d] / gl; }
            }
            for (int d = 0; d < 3; d++) { dU[1 + 2 * d] += dg[d] / (2.f * e); dU[2 + 2 * d] -= dg[d] / (2.f * e); }
            lp[0] = lp[1] + lp[2] + lp[3] + lp[4];
            for (int s = 0; s < 7; s++) {
                const float du = dU[s];
                if (du == 0.f) continue;
                const fit_pt &o = q[s];
                for (int c = 0; c < 8; c++) {
                    const int a = o.c0[0] + (c >> 2 & 1), b = o.c0[1] + (c >> 1 & 1), ee = o.c0[2] + (c & 1);
                    const float wt = (c >> 2 & 1 ? o.ct[0] : 1.f - o.ct[0]) * (c >> 1 & 1 ? o.ct[1] : 1.f - o.ct[1]) * (c & 1 ? o.ct[2] : 1.f - o.ct[2]);
                    if (wt != 0.f && a < f.lz && b < f.ly && ee < f.lx) atomicAdd(&g_delta[((size_t)a * f.ly + b) * f.lx + ee], du * wt);
                }
                const float dp = -du * o.r / (o.p * o.p);
                atomicAdd(&sp_pitch[o.k0], dp * (1.f - o.kt)); atomicAdd(&sp_off[o.k0], du * (1.f - o.kt));
                if (f.nk > 1) { atomicAdd(&sp_pitch[o.k0 + 1], dp * o.kt); atomicAdd(&sp_off[o.k0 + 1], du * o.kt); }
            }
        }
    }
    for (int k = 0; k < 5; k++) red[k][threadIdx.x] = lp[k];
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&parts[k], red[k][0]);
    for (int k = threadIdx.x; k < f.nk; k += blockDim.x) {
        if (sp_pitch[k] != 0.f) atomicAdd(&g_pitch[k], sp_pitch[k]);
        if (sp_off[k] != 0.f) atomicAdd(&g_offset[k], sp_off[k]);
    }
}
extern "C" void nn_fit_loss(const fit_dev *f, double *parts, float *g_delta, float *g_pitch, float *g_offset) {
    if (f->nk > FIT_MAXK) { fprintf(stderr, "nn_fit_loss: %d knots > %d\n", f->nk, FIT_MAXK); abort(); }
    const size_t n = (size_t)f->D * f->H * f->W;
    fit_loss_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(*f, parts, g_delta, g_pitch, g_offset);
    KCHECK();
}

/* smoothness: L = ws sum_n lap_n^2 (6-neighbour laplacian, neighbours inside the lattice), dL/d delta_n = 2 ws (-deg_n lap_n +
   sum_{m in nbr(n)} lap_m); decay wd sum delta^2 */
__global__ void fit_lap_k(const float *v, int lz, int ly, int lx, float *lap) {
    const size_t n = (size_t)lz * ly * lx, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int a = (int)(i / ((size_t)ly * lx)), b = (int)((i / lx) % ly), c = (int)(i % lx);
    float s = 0.f; const float x = v[i];
    if (a > 0) s += v[i - (size_t)ly * lx] - x;
    if (a < lz - 1) s += v[i + (size_t)ly * lx] - x;
    if (b > 0) s += v[i - lx] - x;
    if (b < ly - 1) s += v[i + lx] - x;
    if (c > 0) s += v[i - 1] - x;
    if (c < lx - 1) s += v[i + 1] - x;
    lap[i] = s;
}
__global__ void fit_smooth_k(const float *v, const float *lap, int lz, int ly, int lx, float ws, float wd, double *parts, float *g) {
    __shared__ double red[FIT_TPB];
    const size_t n = (size_t)lz * ly * lx, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    double l = 0;
    if (i < n) {
        const int a = (int)(i / ((size_t)ly * lx)), b = (int)((i / lx) % ly), c = (int)(i % lx);
        int deg = 0; float s = 0.f;
        if (a > 0) { s += lap[i - (size_t)ly * lx]; deg++; }
        if (a < lz - 1) { s += lap[i + (size_t)ly * lx]; deg++; }
        if (b > 0) { s += lap[i - lx]; deg++; }
        if (b < ly - 1) { s += lap[i + lx]; deg++; }
        if (c > 0) { s += lap[i - 1]; deg++; }
        if (c < lx - 1) { s += lap[i + 1]; deg++; }
        g[i] += 2.f * ws * (s - deg * lap[i]) + 2.f * wd * v[i];
        l = (double)ws * lap[i] * lap[i] + (double)wd * v[i] * v[i];
    }
    red[threadIdx.x] = l;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&parts[5], red[0]);
}
extern "C" void nn_fit_smooth(const float *delta, int lz, int ly, int lx, float w_smooth, float w_decay, float *lap, double *parts, float *g_delta) {
    const size_t n = (size_t)lz * ly * lx;
    fit_lap_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lz, ly, lx, lap);
    fit_smooth_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lap, lz, ly, lx, w_smooth, w_decay, parts, g_delta);
    KCHECK();
}

/* predict --fit-fields: pool one window's interior into the fit-field accumulators of the box (one CUDA block per field voxel,
   h^3 threads). acc: 12 planes over the field grid F[3] = ceil(box / h): 0 voxels seen, 1 material (CT > 0), 2-3 material sums of
   2 sigma - 1 of the phase cos / sin channels, 4-9 of the decoded n n^T (diagonal sigma, off-diagonal sigma - 0.5), 10 of the
   density rho = sigma / 32, 11 the max recto probability (float bits, atomicMax as int: non-negative). The window covers box
   voxels [t + halo, t + W - halo) (t: the window origin relative to the box, may be negative), clipped to the box. */
__global__ void fit_accum_k(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
                            int fz0, int fy0, int fx0, int nfy, int nfx, int F1, int F2, size_t nf, int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, int dcode, float *acc) {
    __shared__ float red[11][512];
    const int fi = blockIdx.x, fz = fz0 + fi / (nfy * nfx), fy = fy0 + (fi / nfx) % nfy, fx = fx0 + fi % nfx;
    float v[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; float rmax = 0.f;
    const size_t w3 = (size_t)D * W * W;   /* a D x W x W window, interior planes [hz, D - hz) */
    for (int t = threadIdx.x; t < h * h * h; t += blockDim.x) {   /* h^3 voxels of the field cell over the block's threads */
        const int lz = t / (h * h), ly = (t / h) % h, lx = t % h;
        const int gz = fz * h + lz, gy = fy * h + ly, gx = fx * h + lx;   /* box coordinates */
        const int wz = gz - tz, wy = gy - ty, wx = gx - tx;              /* window coordinates */
        if (gz >= bz || gy >= by || gx >= bx || wz < hz || wz >= D - hz || wy < halo || wy >= W - halo || wx < halo || wx >= W - halo) continue;
        const size_t i = ((size_t)wz * W + wy) * W + wx;
        v[0] += 1.f;
        if (!ct[i]) continue;
        auto sg = [&](int c) { return 1.f / (1.f + __expf(-lg[(size_t)c * w3 + i])); };
        v[1] += 1.f; v[2] += 2.f * sg(ch_phase) - 1.f; v[3] += 2.f * sg(ch_phase + 1) - 1.f;
        for (int k = 0; k < 6; k++) v[4 + k] += sg(ch_normal + k) - (k < 3 ? 0.f : 0.5f);
        { const float sd = sg(ch_density); v[10] += (dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / dscale; }   /* the density code back to rho */
        rmax = fmaxf(rmax, sg(ch_recto));
    }
    for (int k = 0; k < 11; k++) red[k][threadIdx.x] = v[k];
    __shared__ float rm[512]; rm[threadIdx.x] = rmax;
    __syncthreads();
    for (int o = 256; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o && threadIdx.x + o < blockDim.x) { for (int k = 0; k < 11; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; rm[threadIdx.x] = fmaxf(rm[threadIdx.x], rm[threadIdx.x + o]); }
        __syncthreads();
    }
    if (threadIdx.x == 0 && red[0][0] > 0.f) {
        const size_t base = ((size_t)fz * F1 + fy) * F2 + fx;
        for (int k = 0; k < 11; k++) if (red[k][0] != 0.f) atomicAdd(&acc[k * nf + base], red[k][0]);
        atomicMax((int *)&acc[11 * nf + base], __float_as_int(rm[0]));
    }
}
/* the same with a thread per field cell (h <= 4: at h 1 a block per cell ran one thread beside 22.5 KB of shared memory, a few
   threads per SM: ~0.8 s of a 384^3 window whose forward takes 0.25 s); a cell's voxels summed in order (h 1: the same sums) */
__global__ void fit_accum_cell_k(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
                                 int fz0, int fy0, int fx0, int nfz, int nfy, int nfx, int F1, int F2, size_t nf, int ch_recto, int ch_phase, int ch_normal,
                                 int ch_density, float dscale, int dcode, float *acc) {
    const size_t ci = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (ci >= (size_t)nfz * nfy * nfx) return;
    const int fz = fz0 + (int)(ci / ((size_t)nfy * nfx)), fy = fy0 + (int)(ci / nfx % nfy), fx = fx0 + (int)(ci % nfx);
    float v[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; float rmax = 0.f;
    const size_t w3 = (size_t)D * W * W;
    for (int t = 0; t < h * h * h; t++) {
        const int lz = t / (h * h), ly = (t / h) % h, lx = t % h;
        const int gz = fz * h + lz, gy = fy * h + ly, gx = fx * h + lx, wz = gz - tz, wy = gy - ty, wx = gx - tx;
        if (gz >= bz || gy >= by || gx >= bx || wz < hz || wz >= D - hz || wy < halo || wy >= W - halo || wx < halo || wx >= W - halo) continue;
        const size_t i = ((size_t)wz * W + wy) * W + wx;
        v[0] += 1.f;
        if (!ct[i]) continue;
        auto sg = [&](int c) { return 1.f / (1.f + __expf(-lg[(size_t)c * w3 + i])); };
        v[1] += 1.f; v[2] += 2.f * sg(ch_phase) - 1.f; v[3] += 2.f * sg(ch_phase + 1) - 1.f;
        for (int k = 0; k < 6; k++) v[4 + k] += sg(ch_normal + k) - (k < 3 ? 0.f : 0.5f);
        { const float sd = sg(ch_density); v[10] += (dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / dscale; }
        rmax = fmaxf(rmax, sg(ch_recto));
    }
    if (v[0] > 0.f) {
        const size_t base = ((size_t)fz * F1 + fy) * F2 + fx;
        for (int k = 0; k < 11; k++) if (v[k] != 0.f) atomicAdd(&acc[k * nf + base], v[k]);
        atomicMax((int *)&acc[11 * nf + base], __float_as_int(rmax));
    }
}
/* fit-field accumulators (12 planes of nf cells) -> the 10 uint8 fit-field planes (predict's fit_finalize on the device) */
__global__ void fit_finalize_k(const float *acc, size_t nf, int h, uint8_t *a) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= nf) return;
    const float cnt = acc[i], mat = acc[nf + i];
    uint8_t o[10] = {128, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    if (cnt > 0 && mat > 0) {
        /* IEEE divisions (__fdiv_rn: the build's fast math would round some cells differently from the host's fit_finalize) */
        o[9] = (uint8_t)lroundf(255.f * fminf(1.f, __fdiv_rn(mat, cnt)));
        o[0] = (uint8_t)lroundf(128.f + __fdiv_rn(127.f * acc[2 * nf + i], mat)); o[1] = (uint8_t)lroundf(128.f + __fdiv_rn(127.f * acc[3 * nf + i], mat));
        for (int k = 0; k < 6; k++) o[2 + k] = (uint8_t)lroundf(255.f * fminf(1.f, fmaxf(0.f, (k < 3 ? 0.f : 0.5f) + __fdiv_rn(acc[(4 + k) * nf + i], mat))));
        const float rho = __fdiv_rn(acc[10 * nf + i], mat);
        if (rho > 0) { const float v = __fdiv_rn(8.f, __fmul_rn((float)h, rho)); o[8] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lroundf(v)); }
    }
    for (int k = 0; k < 10; k++) a[k * nf + i] = o[k];
}
extern "C" void nn_fit_finalize(const float *acc, size_t nf, int h, uint8_t *a) { fit_finalize_k<<<nblk(nf, 256), 256>>>(acc, nf, h, a); KCHECK(); }
extern "C" void nn_fit_accum_d(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                               int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, int dcode, float *acc) {
    int f0[3], f1[3];
    const int t[3] = {tz, ty, tx}, wn[3] = {D, W, W}, hn[3] = {hz, halo, halo};
    for (int d = 0; d < 3; d++) {
        int lo = t[d] + hn[d], hi = t[d] + wn[d] - hn[d]; if (lo < 0) lo = 0; if (hi > box[d]) hi = box[d];
        f0[d] = lo / h; f1[d] = (hi + h - 1) / h; if (f1[d] > F[d]) f1[d] = F[d];
        if (f1[d] <= f0[d]) return;
    }
    const int nfz = f1[0] - f0[0], nfy = f1[1] - f0[1], nfx = f1[2] - f0[2];
    const size_t nf = (size_t)F[0] * F[1] * F[2];
    if (h > 32 || nf > 0x7fffffff) { fprintf(stderr, "nn_fit_accum: h %d or field grid too large\n", h); abort(); }
    if (h <= 4) fit_accum_cell_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(lg, ct, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                             nfz, nfy, nfx, F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    else fit_accum_k<<<nfz * nfy * nfx, h * h * h < 512 ? h * h * h : 512>>>(lg, ct, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2], nfy, nfx,
                                                     F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    KCHECK();
}
extern "C" void nn_fit_accum(const float *lg, const uint8_t *ct, int W, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                             int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, float *acc) {
    nn_fit_accum_d(lg, ct, W, W, halo, halo, tz, ty, tx, box, h, F, ch_recto, ch_phase, ch_normal, ch_density, dscale, 0, acc);
}

/* analytic data terms (nn_fit_loss2): one evaluation per voxel */
__device__ __forceinline__ unsigned fit_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__global__ void fit_loss2_k(fit_dev f, double *parts, float *g_delta) {
    __shared__ double red[5][FIT_TPB];
    double lp[5] = {0, 0, 0, 0, 0};
    const size_t n = (size_t)f.D * f.H * f.W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const float wv = i < n && (f.stride <= 1 || fit_hash((unsigned)i ^ f.salt) % (unsigned)f.stride == 0) ? f.w[i] * (1.f / 255.f) : 0.f;
    if (wv > 0.f) {
        const int z = (int)(i / ((size_t)f.H * f.W)), y = (int)((i / f.W) % f.H), x = (int)(i % f.W);
        const float pz = (z + 0.5f) * f.h, py = (y + 0.5f) * f.h, px = (x + 0.5f) * f.h, sc = (float)(f.stride > 1 ? f.stride : 1);
        /* axis and its z slope */
        float az = fminf(fmaxf(pz, 0.f), (float)(f.naxis - 1)); int ai = min((int)az, f.naxis - 2 < 0 ? 0 : f.naxis - 2); const float at = f.naxis > 1 ? az - ai : 0.f;
        const int ai1 = f.naxis > 1 ? ai + 1 : ai;
        const float cy = f.cy[ai] + at * (f.cy[ai1] - f.cy[ai]), cx = f.cx[ai] + at * (f.cx[ai1] - f.cx[ai]), cyz = f.cy[ai1] - f.cy[ai], cxz = f.cx[ai1] - f.cx[ai];
        const float dy = py - cy, dx = px - cx, r2 = dy * dy + dx * dx + 1e-6f, r = sqrtf(r2), th = atan2f(dy, dx);
        /* knots (fixed): value and z slope */
        float kz = fminf(fmaxf(pz / (float)f.knot_dz, 0.f), (float)(f.nk - 1)); const int k0 = min((int)kz, f.nk - 2 < 0 ? 0 : f.nk - 2), k1 = f.nk > 1 ? k0 + 1 : k0; const float kt = f.nk > 1 ? kz - k0 : 0.f;
        const float pit = f.pitch[k0] + kt * (f.pitch[k1] - f.pitch[k0]), off = f.offset[k0] + kt * (f.offset[k1] - f.offset[k0]);
        const float pitz = (f.pitch[k1] - f.pitch[k0]) / (float)f.knot_dz, offz = (f.offset[k1] - f.offset[k0]) / (float)f.knot_dz;
        if (r >= f.core * pit) {
            /* trilinear delta, value and gradient */
            const float q[3] = {pz / (float)f.L, py / (float)f.L, px / (float)f.L}; const int dim[3] = {f.lz, f.ly, f.lx}; int c0[3]; float ct[3];
            for (int d = 0; d < 3; d++) { const float a = fminf(fmaxf(q[d], 0.f), (float)(dim[d] - 1)); c0[d] = min((int)a, dim[d] - 2 < 0 ? 0 : dim[d] - 2); ct[d] = dim[d] > 1 ? a - c0[d] : 0.f; }
            float del = 0.f, gdel[3] = {0, 0, 0}, wc[8], gw[8][3]; size_t ic[8];
            const float iL = 1.f / (float)f.L;
            for (int c = 0; c < 8; c++) {
                const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
                const float wz = bz ? ct[0] : 1.f - ct[0], wy = by ? ct[1] : 1.f - ct[1], wx = bx ? ct[2] : 1.f - ct[2];
                wc[c] = wz * wy * wx;
                gw[c][0] = (bz ? iL : -iL) * wy * wx; gw[c][1] = wz * (by ? iL : -iL) * wx; gw[c][2] = wz * wy * (bx ? iL : -iL);
                ic[c] = ((size_t)min(c0[0] + bz, f.lz - 1) * f.ly + min(c0[1] + by, f.ly - 1)) * f.lx + min(c0[2] + bx, f.lx - 1);
                const float v = f.delta[ic[c]];
                del += wc[c] * v; for (int d = 0; d < 3; d++) gdel[d] += gw[c][d] * v;
            }
            const float i2p = 0.15915494309189535f, sdir = (float)f.dir;
            const float drz = -(dy * cyz + dx * cxz) / r;
            float g[3] = {drz / pit - r * pitz / (pit * pit) + offz + gdel[0], dy / (r * pit) + gdel[1], dx / (r * pit) + gdel[2]};
            const float thz = (-dx * cyz + dy * cxz) / r2, thy = dx / r2, thx = -dy / r2;
            g[0] -= sdir * thz * i2p; g[1] -= sdir * thy * i2p; g[2] -= sdir * thx * i2p;
            const float W0 = r / pit + off + del - sdir * th * i2p;
            float dW = 0.f, dg[3] = {0, 0, 0};
            const float mc = ((int)f.pc[i] - 128) * (1.f / 127.f), ms = ((int)f.ps[i] - 128) * (1.f / 127.f), conf = sqrtf(mc * mc + ms * ms);
            if (conf > 0.05f && f.w_phase > 0.f) {
                const float a = 6.283185307179586f * W0 - atan2f(ms, mc); float sa, ca; sincosf(a, &sa, &ca);
                const float cc = f.w_phase * wv * fminf(conf, 1.f) * sc * fit_packed(f, i);
                lp[1] += cc * (1.f - ca); dW += cc * 6.283185307179586f * sa;
            }
            const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + 1e-12f, gl = sqrtf(g2);
            const size_t pl = n;
            const float tzz = f.t[i] * (1.f / 255.f), tyy = f.t[pl + i] * (1.f / 255.f), txx = f.t[2 * pl + i] * (1.f / 255.f);
            const float tzy = f.t[3 * pl + i] * (1.f / 255.f) - 0.5f, tzx = f.t[4 * pl + i] * (1.f / 255.f) - 0.5f, tyx = f.t[5 * pl + i] * (1.f / 255.f) - 0.5f;
            const float tr = tzz + tyy + txx;
            if (tr > 0.1f && f.w_normal > 0.f) {
                const float Tg[3] = {tzz * g[0] + tzy * g[1] + tzx * g[2], tzy * g[0] + tyy * g[1] + tyx * g[2], tzx * g[0] + tyx * g[1] + txx * g[2]};
                const float gTg = g[0] * Tg[0] + g[1] * Tg[1] + g[2] * Tg[2], cc = f.w_normal * wv * sc;
                lp[2] += cc * (1.f - gTg / (g2 * tr));
                for (int d = 0; d < 3; d++) dg[d] += -cc * (2.f * Tg[d] / (g2 * tr) - 2.f * gTg * g[d] / (g2 * g2 * tr));
            }
            if (f.sp[i]) {
                const float rho = 8.f / (f.h * (float)f.sp[i]), lr = logf(gl) - logf(rho);
                if (f.w_density > 0.f) { const float cc = f.w_density * wv * sc; lp[3] += cc * lr * lr; for (int d = 0; d < 3; d++) dg[d] += cc * 2.f * lr * g[d] / g2; }
                const float lo = 0.4f * rho - gl;
                if (lo > 0.f && f.w_nondeg > 0.f) { const float cc = f.w_nondeg * wv * sc; lp[4] += cc * lo * lo; for (int d = 0; d < 3; d++) dg[d] += -cc * 2.f * lo * g[d] / gl; }
            }
            lp[0] = lp[1] + lp[2] + lp[3] + lp[4];
            for (int c = 0; c < 8; c++) {
                const float gv = dW * wc[c] + dg[0] * gw[c][0] + dg[1] * gw[c][1] + dg[2] * gw[c][2];
                if (gv != 0.f) atomicAdd(&g_delta[ic[c]], gv);
            }
        }
    }
    for (int k = 0; k < 5; k++) red[k][threadIdx.x] = lp[k];
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&parts[k], red[k][0]);
}
extern "C" void nn_fit_loss2(const fit_dev *f, double *parts, float *g_delta) {
    const size_t n = (size_t)f->D * f->H * f->W;
    fit_loss2_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(*f, parts, g_delta);
    KCHECK();
}

__global__ void fit_grid_k(const fit_dev *ms, const fit_grid_row *rows, int ny, int nx, float y0, float x0, float step, int with_theta, float *out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; const int a = blockIdx.y;
    if (i >= (size_t)ny * nx) return;
    const int b = (int)(i / nx), c = (int)(i % nx);
    const fit_grid_row r = rows[a];
    const float y = y0 + ((float)b + 0.5f) * step, x = x0 + ((float)c + 0.5f) * step, itp = 0.15915494309189535f;
    fit_pt p; fit_eval(ms[r.a], r.z - r.za, y, x, p);
    const float tha = p.th;
    float W = p.U - (float)ms[r.a].dir * p.th * itp;
    if (r.t > 0.f) { fit_pt q; fit_eval(ms[r.b], r.z - r.zb, y, x, q); W = (1.f - r.t) * W + r.t * (q.U - (float)ms[r.b].dir * q.th * itp); }
    out[(size_t)a * ny * nx + i] = with_theta ? W + (float)ms[r.a].dir * tha * itp : W;
}
void nn_fit_grid(const fit_dev *ms, const fit_grid_row *rows, int nz, int ny, int nx, float y0, float x0, float step, int with_theta, float *out) {
    if (nz <= 0 || ny <= 0 || nx <= 0) return;
    const size_t per = (size_t)ny * nx;
    for (int z0 = 0; z0 < nz; z0 += 65535) {
        const int n = nz - z0 < 65535 ? nz - z0 : 65535;
        fit_grid_k<<<dim3((unsigned)((per + 255) / 256), (unsigned)n), 256>>>(ms, rows + z0, ny, nx, y0, x0, step, with_theta, out + (size_t)z0 * per);
    }
    CK(cudaGetLastError());
}
