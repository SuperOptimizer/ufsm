/* CUDA kernels of the winding fitter (src/fit.c, src/fit.h). Per field voxel: U at the centre and at +-h/2 along each axis
   (U = r / pitch(z) + offset(z) + trilinear delta), W = U - dir theta / 2 pi, grad W by central differences with the theta
   difference wrapped (so the branch cut never enters), the data terms, and dL/dU at the seven points scattered to the lattice
   corners and the knots (knots through per-block shared sums). */
#include "nn_common.cuh"
extern "C" {
#include "fit.h"
}
#include "fit_terms.h"

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
__device__ __forceinline__ float fit_wrap(float a) {   /* into (-pi, pi] */
    const float tp = 6.283185307179586f;
    a -= tp * floorf((a + 3.14159265358979f) / tp);
    return a;
}

/* a block's per-thread loss parts summed into parts (every FIT_P_* slot but the smoothness, which the lattice kernel owns) */
__device__ __forceinline__ void fit_reduce_parts(const double *lp, double (*red)[FIT_TPB], double *parts) {
    for (int k = 0; k < FIT_NPARTS; k++) red[k][threadIdx.x] = lp[k];
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < FIT_NPARTS; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH && red[k][0] != 0.0) atomicAdd(&parts[k], red[k][0]);
}
/* Deterministic sums (2026-10-26): the gradient and the loss parts accumulate as 64-bit fixed-point integers (integer adds commute,
   so the result does not depend on the block / list order; the float / double atomics before made every GPU fit differ in the last
   bits from run to run, which the chaotic slab fits amplified to ~0.05 of joined ids). Gradient: FIT_GQ = 2^50 per unit (|node sum|
   < 8192, resolution 9e-16; per-voxel contributions are ~1e-7..1e-4), parts: FIT_PQ = 2^40 (|part| < 8e6). Converted once per call. */
#define FIT_GQ 1125899906842624.0
#define FIT_PQ 1099511627776.0
__device__ __forceinline__ void fit_gadd(long long *gi, size_t k, float v) { atomicAdd((unsigned long long *)&gi[k], (unsigned long long)__double2ll_rn((double)v * FIT_GQ)); }
__device__ __forceinline__ long long fit_pfix(double v) { return __double2ll_rn(v * FIT_PQ); }
/* a block's per-thread fixed-point parts summed (integers: any order) into pi */
__device__ __forceinline__ void fit_reduce_parts_fix(const long long *lp, long long (*red)[FIT_TPB], long long *pi) {
    for (int k = 0; k < FIT_NPARTS; k++) red[k][threadIdx.x] = lp[k];
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < FIT_NPARTS; k++) red[k][threadIdx.x] += red[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < FIT_NPARTS; k++) if (k != FIT_P_SMOOTH && red[k][0] != 0) atomicAdd((unsigned long long *)&pi[k], (unsigned long long)red[k][0]);
}
__global__ void fit_gconv_k(const long long *gi, size_t n, float *g) {   /* g += gi / FIT_GQ */
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) if (gi[i]) g[i] += (float)((double)gi[i] * (1.0 / FIT_GQ));
}
__global__ void fit_pconv_k(const long long *pi, double *parts) { if (threadIdx.x < FIT_NPARTS && pi[threadIdx.x]) parts[threadIdx.x] += (double)pi[threadIdx.x] * (1.0 / FIT_PQ); }
static long long *g_fix_g, *g_fix_p; static size_t g_fix_gn;
static long long *fit_fix_scratch(size_t n) {   /* the gradient scratch (n nodes) and the parts' (zeroed) */
    if (!g_fix_p) CK(cudaMalloc(&g_fix_p, 16 * sizeof(long long)));
    if (n > g_fix_gn) { if (g_fix_g) CK(cudaFree(g_fix_g)); CK(cudaMalloc(&g_fix_g, n * sizeof(long long))); g_fix_gn = n; }
    CK(cudaMemsetAsync(g_fix_g, 0, n * sizeof(long long))); CK(cudaMemsetAsync(g_fix_p, 0, 16 * sizeof(long long)));
    return g_fix_g;
}
__global__ void fit_loss_k(fit_dev f, double *parts, float *g_delta, float *g_pitch, float *g_offset) {
    __shared__ float sp_pitch[FIT_MAXK], sp_off[FIT_MAXK];
    __shared__ double red[FIT_NPARTS][FIT_TPB];
    for (int k = threadIdx.x; k < f.nk; k += blockDim.x) { sp_pitch[k] = 0.f; sp_off[k] = 0.f; }
    __syncthreads();
    double lp[FIT_NPARTS] = {0};
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
            fit_phase_term(&f, i, W0, wv, &lp[FIT_P_PHASE], &dU[0]);
            fit_recto_term(&f, i, W0, wv, &lp[FIT_P_RECTO], &dU[0]);
            fit_ct_term(&f, i, W0, wv, &lp[FIT_P_CT], &dU[0]);
            fit_grad_terms(&f, i, g, wv, lp, dg);
            fit_pgrad_term(&f, i, g, wv, lp, dg);
            fit_cspace_term(&f, i, g, wv, lp, dg);
            for (int d = 0; d < 3; d++) { dU[1 + 2 * d] += dg[d] / (2.f * e); dU[2 + 2 * d] -= dg[d] / (2.f * e); }
            lp[FIT_P_TOTAL] = lp[FIT_P_PHASE] + lp[FIT_P_NORMAL] + lp[FIT_P_DENSITY] + lp[FIT_P_NONDEG] + lp[FIT_P_PGRAD] + lp[FIT_P_RECTO] + lp[FIT_P_CT] + lp[FIT_P_CSPACE];
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
    fit_reduce_parts(lp, red, parts);
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
/* g += 2 ws (sum_{m in nbr(n)} wl_m - deg_n wl_n) + 2 wd v_n with wl = wn lap (the per-node weighted laplacian; wn null: 1); the loss
   ws wn lap^2 + wd v^2 */
__global__ void fit_smooth_k(const float *v, const float *lap, const float *wn, int lz, int ly, int lx, float ws, float wd, long long *pi, float *g) {
    __shared__ double red[FIT_TPB];
    const size_t n = (size_t)lz * ly * lx, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    double l = 0;
    if (i < n) {
        const int a = (int)(i / ((size_t)ly * lx)), b = (int)((i / lx) % ly), c = (int)(i % lx);
#define WL(j_) (wn ? wn[j_] * lap[j_] : lap[j_])
        int deg = 0; float s = 0.f;
        if (a > 0) { s += WL(i - (size_t)ly * lx); deg++; }
        if (a < lz - 1) { s += WL(i + (size_t)ly * lx); deg++; }
        if (b > 0) { s += WL(i - lx); deg++; }
        if (b < ly - 1) { s += WL(i + lx); deg++; }
        if (c > 0) { s += WL(i - 1); deg++; }
        if (c < lx - 1) { s += WL(i + 1); deg++; }
        const float wli = WL(i);
#undef WL
        g[i] += 2.f * ws * (s - deg * wli) + 2.f * wd * v[i];
        l = (double)ws * wli * lap[i] + (double)wd * v[i] * v[i];
    }
    red[threadIdx.x] = l;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0 && red[0] != 0.0) atomicAdd((unsigned long long *)&pi[6], (unsigned long long)fit_pfix(red[0]));   /* (fixed point: deterministic) */
}
extern "C" void nn_fit_smooth(const float *delta, int lz, int ly, int lx, float w_smooth, float w_decay, const float *wn, float *lap, double *parts, float *g_delta) {
    const size_t n = (size_t)lz * ly * lx;
    fit_lap_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lz, ly, lx, lap);
    static long long *ps; if (!ps) CK(cudaMalloc(&ps, 16 * sizeof(long long)));
    CK(cudaMemsetAsync(ps, 0, 16 * sizeof(long long)));
    fit_smooth_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lap, wn, lz, ly, lx, w_smooth, w_decay, ps, g_delta);
    fit_pconv_k<<<1, 32>>>(ps, parts);
    KCHECK();
}

/* predict --fit-fields: pool one window's interior into the fit-field accumulators of the box (one CUDA block per field voxel,
   h^3 threads). acc: 12 planes over the field grid F[3] = ceil(box / h): 0 voxels seen, 1 material (CT > 0), 2-3 material sums of
   2 sigma - 1 of the phase cos / sin channels, 4-9 of the decoded n n^T (diagonal sigma, off-diagonal sigma - 0.5), 10 of the
   density rho = sigma / 32, 11 the max recto probability (float bits, atomicMax as int: non-negative). The window covers box
   voxels [t + halo, t + W - halo) (t: the window origin relative to the box, may be negative), clipped to the box. */
template <typename LT>
__global__ void fit_accum_k(const LT *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
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
        if (!(bits ? (bits[i >> 5] >> (i & 31)) & 1u : ct[i])) continue;   /* material: the uint8 CT window or its bit plane */
        auto sg = [&](int c) { return 1.f / (1.f + __expf(-lgv(lg, (size_t)c * w3 + i))); };
        v[1] += 1.f; v[2] += 2.f * sg(ch_phase) - 1.f; v[3] += 2.f * sg(ch_phase + 1) - 1.f;
        for (int k = 0; k < 6; k++) v[4 + k] += sg(ch_normal + k) - (k < 3 ? 0.f : 0.5f);
        { const float sd = sg(ch_density); v[10] += (dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / dscale; }   /* the density code back to rho */
        if (ch_recto >= 0) rmax = fmaxf(rmax, sg(ch_recto));   /* (task unwrap2: no recto head) */
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
template <typename LT>
__global__ void fit_accum_cell_k(const LT *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
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
        if (!(bits ? (bits[i >> 5] >> (i & 31)) & 1u : ct[i])) continue;   /* material: the uint8 CT window or its bit plane */
        auto sg = [&](int c) { return 1.f / (1.f + __expf(-lgv(lg, (size_t)c * w3 + i))); };
        v[1] += 1.f; v[2] += 2.f * sg(ch_phase) - 1.f; v[3] += 2.f * sg(ch_phase + 1) - 1.f;
        for (int k = 0; k < 6; k++) v[4 + k] += sg(ch_normal + k) - (k < 3 ? 0.f : 0.5f);
        { const float sd = sg(ch_density); v[10] += (dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / dscale; }
        if (ch_recto >= 0) rmax = fmaxf(rmax, sg(ch_recto));   /* (task unwrap2: no recto head) */
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
    nn_fit_accum_dm(lg, ct, nullptr, D, W, hz, halo, tz, ty, tx, box, h, F, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
}
/* the same with the material mask as the CT window's bit plane (nn_pred_ctmask; ct unused): predict keeps the uint8 window in the
   network's skip buffer (inference aliasing), which the forward overwrites */
extern "C" void nn_fit_accum_dm(const float *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
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
    const f16 *lgh = (const f16 *)lg;   /* nn_set_logits_h16: fp16 logits */
    if (h <= 4 && g_logits_h16) fit_accum_cell_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(lgh, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                             nfz, nfy, nfx, F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    else if (h > 4 && g_logits_h16) fit_accum_k<<<nfz * nfy * nfx, h * h * h < 512 ? h * h * h : 512>>>(lgh, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2], nfy, nfx,
                                                     F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    else if (h <= 4) fit_accum_cell_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(lg, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                             nfz, nfy, nfx, F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    else fit_accum_k<<<nfz * nfy * nfx, h * h * h < 512 ? h * h * h : 512>>>(lg, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2], nfy, nfx,
                                                     F[1], F[2], nf, ch_recto, ch_phase, ch_normal, ch_density, dscale, dcode, acc);
    KCHECK();
}
extern "C" void nn_fit_accum(const float *lg, const uint8_t *ct, int W, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                             int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, float *acc) {
    nn_fit_accum_d(lg, ct, W, W, halo, halo, tz, ty, tx, box, h, F, ch_recto, ch_phase, ch_normal, ch_density, dscale, 0, acc);
}

/* the recto plane: accumulator plane 11 (the max recto probability over the cell's material voxels) x255 */
__global__ void fit_finalize_recto_k(const float *acc, size_t nf, uint8_t *a) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= nf) return;
    a[i] = (uint8_t)lroundf(255.f * fminf(1.f, fmaxf(0.f, acc[11 * nf + i])));
}
extern "C" void nn_fit_finalize_recto(const float *acc, size_t nf, uint8_t *a) { fit_finalize_recto_k<<<nblk(nf, 256), 256>>>(acc, nf, a); KCHECK(); }

/* predict --fit-ct: the CT planes of the fit fields. (a) nn_fit_accum_ct: per cell the sum of the window's uint8 CT over the cell's voxels
   in the window interior (the same voxels as nn_fit_accum's count plane 0), into acc_ct; call it while the CT window is intact. (b)
   nn_fit_accum_cs: per cell whose centre voxel lies in this window's interior, the CT sampled along the network normal there (the top
   eigenvector of the decoded n n^T) over +-2.5 x the network spacing (60..400 um, clipped to the window), its papyrus peaks counted
   (src/fit_terms.h fit_ct_spacing, prominence 1 sd): the spacing in native voxels stored into acc_cs (0: none). */
__global__ void fit_accum_ct_k(const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
                               int fz0, int fy0, int fx0, int nfz, int nfy, int nfx, int F1, int F2, float *acc) {
    const size_t ci = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (ci >= (size_t)nfz * nfy * nfx) return;
    const int fz = fz0 + (int)(ci / ((size_t)nfy * nfx)), fy = fy0 + (int)(ci / nfx % nfy), fx = fx0 + (int)(ci % nfx);
    float v = 0.f;
    for (int t = 0; t < h * h * h; t++) {
        const int lz = t / (h * h), ly = (t / h) % h, lx = t % h;
        const int gz = fz * h + lz, gy = fy * h + ly, gx = fx * h + lx, wz = gz - tz, wy = gy - ty, wx = gx - tx;
        if (gz >= bz || gy >= by || gx >= bx || wz < hz || wz >= D - hz || wy < halo || wy >= W - halo || wx < halo || wx >= W - halo) continue;
        v += (float)ct[((size_t)wz * W + wy) * W + wx];
    }
    if (v != 0.f) atomicAdd(&acc[((size_t)fz * F1 + fy) * F2 + fx], v);
}
template <typename LT>
__global__ void fit_accum_cs_k(const LT *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
                               int fz0, int fy0, int fx0, int nfz, int nfy, int nfx, int F1, int F2, int ch_normal, int ch_density, float dscale, int dcode, float um, float *acc) {
    const size_t ci = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (ci >= (size_t)nfz * nfy * nfx) return;
    const int fz = fz0 + (int)(ci / ((size_t)nfy * nfx)), fy = fy0 + (int)(ci / nfx % nfy), fx = fx0 + (int)(ci % nfx);
    const int gz = fz * h + h / 2, gy = fy * h + h / 2, gx = fx * h + h / 2, wz = gz - tz, wy = gy - ty, wx = gx - tx;
    if (gz >= bz || gy >= by || gx >= bx || wz < hz || wz >= D - hz || wy < halo || wy >= W - halo || wx < halo || wx >= W - halo) return;
    const size_t w3 = (size_t)D * W * W, i = ((size_t)wz * W + wy) * W + wx;
    auto sg = [&](int c) { return 1.f / (1.f + __expf(-lgv(lg, (size_t)c * w3 + i))); };
    const float tzz = sg(ch_normal), tyy = sg(ch_normal + 1), txx = sg(ch_normal + 2), tzy = sg(ch_normal + 3) - 0.5f, tzx = sg(ch_normal + 4) - 0.5f, tyx = sg(ch_normal + 5) - 0.5f;
    float out = 0.f;
    if (ct[i] && tzz + tyy + txx > 0.1f) {
        float v0 = 0.31f, v1 = 0.57f, v2 = 0.76f;
        for (int it = 0; it < 30; it++) {
            const float u0 = tzz * v0 + tzy * v1 + tzx * v2, u1 = tzy * v0 + tyy * v1 + tyx * v2, u2 = tzx * v0 + tyx * v1 + txx * v2, l = sqrtf(u0 * u0 + u1 * u1 + u2 * u2);
            if (l < 1e-9f) break; v0 = u0 / l; v1 = u1 / l; v2 = u2 / l;
        }
        const float sd = sg(ch_density), rho = (dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / dscale;   /* turns per native voxel */
        float R = rho > 0.f ? 2.5f / rho * um : 200.f; R = fminf(fmaxf(R, 60.f), 400.f); R /= um;   /* native voxels */
        /* clip the ray to the window: the largest symmetric extent along +-v inside [0, D - 1] x [0, W - 1]^2 */
        const float pc[3] = {(float)wz, (float)wy, (float)wx}, vv[3] = {v0, v1, v2}, hi[3] = {(float)(D - 1), (float)(W - 1), (float)(W - 1)};
        for (int d = 0; d < 3; d++) if (fabsf(vv[d]) > 1e-6f) { const float a = fminf(pc[d], hi[d] - pc[d]) / fabsf(vv[d]); R = fminf(R, a); }
        float p[256]; const float stp = fmaxf(1.f, 2.f * R / 255.f); const int n = 2 * (int)(R / stp) + 1;
        for (int k = 0; k < n; k++) {
            const float t = (k - n / 2) * stp, q[3] = {pc[0] + t * v0, pc[1] + t * v1, pc[2] + t * v2};
            int i0[3]; float tt[3];
            for (int d = 0; d < 3; d++) { const float a = fminf(fmaxf(q[d], 0.f), hi[d]); i0[d] = min((int)a, (int)hi[d] - 1 < 0 ? 0 : (int)hi[d] - 1); tt[d] = a - i0[d]; }
            float s = 0.f;
            for (int c = 0; c < 8; c++) {
                const int a = i0[0] + (c >> 2 & 1), b = i0[1] + (c >> 1 & 1), e = i0[2] + (c & 1);
                const float w = (c >> 2 & 1 ? tt[0] : 1.f - tt[0]) * (c >> 1 & 1 ? tt[1] : 1.f - tt[1]) * (c & 1 ? tt[2] : 1.f - tt[2]);
                if (w != 0.f) s += w * (float)ct[((size_t)a * W + b) * W + e];
            }
            p[k] = s;
        }
        out = fit_ct_spacing(p, n, stp, 1.f);
    }
    acc[((size_t)fz * F1 + fy) * F2 + fx] = out;
}
static int fit_ct_range(int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3], int f0[3], int f1[3]) {
    const int t[3] = {tz, ty, tx}, wn[3] = {D, W, W}, hn[3] = {hz, halo, halo};
    for (int d = 0; d < 3; d++) {
        int lo = t[d] + hn[d], hi = t[d] + wn[d] - hn[d]; if (lo < 0) lo = 0; if (hi > box[d]) hi = box[d];
        f0[d] = lo / h; f1[d] = (hi + h - 1) / h; if (f1[d] > F[d]) f1[d] = F[d];
        if (f1[d] <= f0[d]) return 0;
    }
    return 1;
}
extern "C" void nn_fit_accum_ct(const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3], float *acc_ct) {
    int f0[3], f1[3]; if (!fit_ct_range(D, W, hz, halo, tz, ty, tx, box, h, F, f0, f1)) return;
    const int nfz = f1[0] - f0[0], nfy = f1[1] - f0[1], nfx = f1[2] - f0[2];
    fit_accum_ct_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(ct, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2], nfz, nfy, nfx, F[1], F[2], acc_ct);
    KCHECK();
}
extern "C" void nn_fit_accum_cs(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                                int ch_normal, int ch_density, float dscale, int dcode, float um, float *acc_cs) {
    int f0[3], f1[3]; if (!fit_ct_range(D, W, hz, halo, tz, ty, tx, box, h, F, f0, f1)) return;
    const int nfz = f1[0] - f0[0], nfy = f1[1] - f0[1], nfx = f1[2] - f0[2];
    if (g_logits_h16) fit_accum_cs_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>((const f16 *)lg, ct, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                                 nfz, nfy, nfx, F[1], F[2], ch_normal, ch_density, dscale, dcode, um, acc_cs);
    else fit_accum_cs_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(lg, ct, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                      nfz, nfy, nfx, F[1], F[2], ch_normal, ch_density, dscale, dcode, um, acc_cs);
    KCHECK();
}
/* the CT planes from the count plane (cnt, accumulator plane 0), the CT sums and the spacings: ct = round(sum / count), cs = the spacing
   in sp units (h / 8 native voxels; 0 unknown) */
__global__ void fit_finalize_ct_k(const float *cnt, const float *sum, const float *spc, size_t nf, int h, uint8_t *oct, uint8_t *ocs) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= nf) return;
    oct[i] = cnt[i] > 0.f ? (uint8_t)fminf(255.f, lroundf(__fdiv_rn(sum[i], cnt[i]))) : 0;
    const float v = spc[i] > 0.f ? spc[i] * 8.f / (float)h : 0.f;
    ocs[i] = v <= 0.f ? 0 : v < 1.f ? 1 : v > 255.f ? 255 : (uint8_t)lroundf(v);
}
extern "C" void nn_fit_finalize_ct(const float *acc, size_t nf, int h, uint8_t *oct, uint8_t *ocs) {
    fit_finalize_ct_k<<<nblk(nf, 256), 256>>>(acc, acc + 12 * nf, acc + 13 * nf, nf, h, oct, ocs); KCHECK();
}

/* predict --fit-fields of a task unwrap2 checkpoint: the winding mod 4 head as gauge-free relative planes. Per field cell and axis a
   (z, y, x) the material-voxel sums of p(v + d e_a) conj p(v), p = (2 sigma - 1) of the cos / sin channels (ch_w4, ch_w4 + 1): the
   predicted W difference mod 4 over d voxels, inside one window (one gauge). acc: 9 planes of nf cells (re, im, count per axis);
   needs d <= halo (y, x) and <= hz (z), so v + d e_a stays inside the window. */
template <typename LT>
__global__ void fit_accum_w4_k(const LT *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, int bz, int by, int bx, int h,
                               int fz0, int fy0, int fx0, int nfz, int nfy, int nfx, int F1, int F2, size_t nf, int ch, int dd, float *acc) {
    const size_t ci = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (ci >= (size_t)nfz * nfy * nfx) return;
    const int fz = fz0 + (int)(ci / ((size_t)nfy * nfx)), fy = fy0 + (int)(ci / nfx % nfy), fx = fx0 + (int)(ci % nfx);
    float v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const size_t w3 = (size_t)D * W * W, st[3] = {(size_t)W * W * dd, (size_t)W * dd, (size_t)dd};
    auto mat = [&](size_t i) { return bits ? (bits[i >> 5] >> (i & 31)) & 1u : (unsigned)(ct[i] != 0); };
    for (int t = 0; t < h * h * h; t++) {
        const int lz = t / (h * h), ly = (t / h) % h, lx = t % h;
        const int gz = fz * h + lz, gy = fy * h + ly, gx = fx * h + lx, wz = gz - tz, wy = gy - ty, wx = gx - tx;
        if (gz >= bz || gy >= by || gx >= bx || wz < hz || wz >= D - hz || wy < halo || wy >= W - halo || wx < halo || wx >= W - halo) continue;
        const size_t i = ((size_t)wz * W + wy) * W + wx;
        if (!mat(i)) continue;
        const float pr = 2.f / (1.f + __expf(-lgv(lg, (size_t)ch * w3 + i))) - 1.f, pi = 2.f / (1.f + __expf(-lgv(lg, (size_t)(ch + 1) * w3 + i))) - 1.f;
        for (int a = 0; a < 3; a++) {
            const size_t j = i + st[a];
            if (!mat(j)) continue;
            const float qr = 2.f / (1.f + __expf(-lgv(lg, (size_t)ch * w3 + j))) - 1.f, qi = 2.f / (1.f + __expf(-lgv(lg, (size_t)(ch + 1) * w3 + j))) - 1.f;
            v[3 * a] += qr * pr + qi * pi; v[3 * a + 1] += qi * pr - qr * pi; v[3 * a + 2] += 1.f;
        }
    }
    const size_t base = ((size_t)fz * F1 + fy) * F2 + fx;
    for (int k = 0; k < 9; k++) if (v[k] != 0.f) atomicAdd(&acc[k * nf + base], v[k]);
}
extern "C" void nn_fit_accum_w4(const float *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                                int ch_w4, int dd, float *acc) {
    if (dd > halo || dd > hz || dd < 1) { fprintf(stderr, "nn_fit_accum_w4: offset %d outside the halo (%d, z %d)\n", dd, halo, hz); abort(); }
    int f0[3], f1[3];
    const int t[3] = {tz, ty, tx}, wn[3] = {D, W, W}, hn[3] = {hz, halo, halo};
    for (int d = 0; d < 3; d++) {
        int lo = t[d] + hn[d], hi = t[d] + wn[d] - hn[d]; if (lo < 0) lo = 0; if (hi > box[d]) hi = box[d];
        f0[d] = lo / h; f1[d] = (hi + h - 1) / h; if (f1[d] > F[d]) f1[d] = F[d];
        if (f1[d] <= f0[d]) return;
    }
    const int nfz = f1[0] - f0[0], nfy = f1[1] - f0[1], nfx = f1[2] - f0[2];
    const size_t nf = (size_t)F[0] * F[1] * F[2];
    if (g_logits_h16) fit_accum_w4_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>((const f16 *)lg, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                                 nfz, nfy, nfx, F[1], F[2], nf, ch_w4, dd, acc);
    else fit_accum_w4_k<<<nblk((size_t)nfz * nfy * nfx, 128), 128>>>(lg, ct, bits, D, W, hz, halo, tz, ty, tx, box[0], box[1], box[2], h, f0[0], f0[1], f0[2],
                                                                      nfz, nfy, nfx, F[1], F[2], nf, ch_w4, dd, acc);
    KCHECK();
}
/* the 9 relative planes -> 4 uint8 planes: dz dy dx = the angle of each axis' sum (byte b: W(x + d e_a) - W(x) = b / 64 turns mod 4),
   dq = 255 x the mean over the axes with pairs of |sum| / pairs (0: no information) */
__global__ void fit_finalize_w4_k(const float *acc, size_t nf, uint8_t *a) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= nf) return;
    float q = 0.f; int na = 0;
    for (int ax = 0; ax < 3; ax++) {
        const float re = acc[(3 * ax) * nf + i], im = acc[(3 * ax + 1) * nf + i], n = acc[(3 * ax + 2) * nf + i];
        uint8_t b = 0;
        if (n > 0.f) { b = (uint8_t)(lrintf(atan2f(im, re) * (256.f / 6.283185307179586f)) & 255); q += fminf(1.f, sqrtf(re * re + im * im) / n); na++; }
        a[ax * nf + i] = b;
    }
    a[3 * nf + i] = na ? (uint8_t)lrintf(255.f * q / na) : 0;
}
extern "C" void nn_fit_finalize_w4(const float *acc, size_t nf, uint8_t *a) { fit_finalize_w4_k<<<nblk(nf, 256), 256>>>(acc, nf, a); KCHECK(); }

/* nn_fit_accum_lat (fit.h): one block of 512 threads per field cell of 8^3 voxels (2^3 latent cells: --fit-zarr's windows keep the field
   cells on the cell grid), a voxel per thread in the sums as fit_accum_k. The plane path (predict's z-slab fzs: nn_lat_up_slab +
   nn_lat_shuffle + nn_fstem_fwd into [11][nz][W][W] fp16 planes, fit_accum_k, then the winding's own planes over +-d and fit_accum_w4_k)
   wrote and read ~30 B of planes per voxel and ran the stem twice; here the values stay in shared memory, with the same operations in
   the same order:
   - the upsampling (lat_up4_k): per (channel, latent cell, z offset) the cell's 3^3 coarse neighbourhood (staged: the 4^3 cells around
     the field cell) z then y then x, fp16 into the voxel planes L; fl_lw pins the multiply-adds to the form the compiler gives
     lat_up4_k's sums (fma(w2, c, fma(w0, a, w1 b)): a plain expression here contracts differently, 1 ulp in ~1e-5 of the voxels);
   - the sub-cell term (lat_shuf_k) and the stem's (fstem_fwd_k's mma sequence, b1, silu, its W2 projection), each added and rounded;
   - the cell's sums: fit_accum_k's per-thread expressions and its shared-memory tree; the winding mod 4's: per voxel the terms of
     fit_accum_w4_k, summed over the cell's voxels in its order by one thread per accumulator (a serial chain there too).
   Regions: 0 the cell's voxels, 1-3 the same shifted by dd (a multiple of 4) along z, y, x: the winding's partner voxels. */
enum { FL_T = 512, FL_MAXCH = 12, FL_MAXS = 4, FL_C = 16 };
__device__ static const float FL_LW[4][3] = {{0.375f, 0.625f, 0.f}, {0.1875f, 0.75f, 0.0625f}, {0.0625f, 0.75f, 0.1875f}, {0.f, 0.625f, 0.375f}};
__device__ __forceinline__ float fl_lw(int r, float a, float b, float c) { return __fmaf_rn(FL_LW[r][2], c, __fmaf_rn(FL_LW[r][0], a, __fmul_rn(FL_LW[r][1], b))); }
__device__ __forceinline__ float fl_h(float v) { return __half2float(__float2half_rn(v)); }
struct fl_args {
    int d, h, w, D, H, W, hz, halo, tz, ty, tx, bz, by, bx, fz0, fy0, fx0, nfy, nfx, F1, F2;
    size_t nf;
    int ch_recto, dcode, ch_w4, dd, sbase, C;
    float dscale, mean, isd;
    int nch, ndm, ch[FL_MAXCH], slot[FL_MAXCH];   /* region 0's channels (ndm of the cell sums, then the winding pair) and their stem slots (-1 none) */
    int nsl, so[FL_MAXS], wso[2];                 /* region 0's slots -> stem outputs; regions 1-3: slot u = the winding channel u's output wso[u] (-1 none) */
    float *dbg;                                   /* (test_fit_lat: region 0's logits of the in-window voxels, [nch][D H W] fp32) */
};
static float *g_fl_dbg;
extern "C" void nn_fit_accum_lat_debug(float *planes) { g_fl_dbg = planes; }
template <typename T>
__global__ void __launch_bounds__(FL_T) fit_accum_lat_k(const T *__restrict__ lg, const float *__restrict__ sp, const uint8_t *__restrict__ ctx, const uint8_t *__restrict__ ct,
                                                       const uint32_t *__restrict__ bits, fl_args A, float *acc, float *acc4) {
    __shared__ float cs0[FL_MAXCH][64], csw[3][2][64], sst[4][64][FL_MAXS];
    __shared__ union { float red[12][FL_T]; float shw[FL_T / 32][8][FL_C + 1]; struct { __half l0[FL_MAXCH][FL_T], lw[3][2][FL_T]; } L; } U;
    const int tid = threadIdx.x, fi = blockIdx.x;
    const int fz = A.fz0 + fi / (A.nfy * A.nfx), fy = A.fy0 + (fi / A.nfx) % A.nfy, fx = A.fx0 + fi % A.nfx;
    const int nreg = A.ch_w4 >= 0 ? 4 : 1, C = A.C;
    int org[4][3];
    for (int r = 0; r < 4; r++) { org[r][0] = fz * 8 - A.tz; org[r][1] = fy * 8 - A.ty; org[r][2] = fx * 8 - A.tx; if (r) org[r][r - 1] += A.dd; }
    const size_t cv = (size_t)A.d * A.h * A.w, hw = (size_t)A.H * A.W;
    /* the coarse neighbourhoods: staged cell q of a region = the clamped cell (o >> 2) - 1 + (q >> 4, q >> 2 & 3, q & 3) */
    {
        const int n0 = A.nch * 64, nt = n0 + (nreg - 1) * 2 * 64;
        for (int k = tid; k < nt; k += FL_T) {
            const int r = k < n0 ? 0 : 1 + (k - n0) / 128, cc = k < n0 ? k >> 6 : (k - n0) >> 6 & 1, q = k & 63;
            const int c = r ? A.ch[A.ndm + cc] : A.ch[cc];
            const int cz = min(max((org[r][0] >> 2) - 1 + (q >> 4), 0), A.d - 1), cy = min(max((org[r][1] >> 2) - 1 + (q >> 2 & 3), 0), A.h - 1), cx = min(max((org[r][2] >> 2) - 1 + (q & 3), 0), A.w - 1);
            const float v = ldv(lg, (size_t)c * cv + ((size_t)cz * A.h + cy) * A.w + cx);
            if (r) csw[r - 1][cc][q] = v; else cs0[cc][q] = v;
        }
    }
    /* the stem's outputs on the 4^3 2^3-blocks of each region (block q: (o >> 1) + (q >> 4, q >> 2 & 3, q & 3)), fstem_fwd_k's sequence */
    if (C) {
        const int lane = tid & 31, wp = tid >> 5, g = lane >> 2, tq = lane & 3;
        unsigned af[4][4];
        for (int s_ = 0; s_ < 4; s_++) for (int r = 0; r < 4; r++) {
            const int row = g + (r & 1) * 8, col = 16 * s_ + 2 * tq + (r >> 1) * 8;
            const float w0 = row < C ? sp[row * 64 + col] : 0.f, w1 = row < C ? sp[row * 64 + col + 1] : 0.f;
            const __half2 h2 = __floats2half2_rn(w0, w1); af[s_][r] = *(const unsigned *)&h2;
        }
        const float *b1 = sp + C * 64, *W2 = sp + C * 64 + C;
        auto tap = [&](int z, int y, int x) -> __half {
            return z >= 0 && z < A.D && y >= 0 && y < A.H && x >= 0 && x < A.W ? __float2half(((float)ctx[(size_t)z * hw + (size_t)y * A.W + x] - A.mean) * A.isd) : __float2half(0.f);
        };
        const int nsr = A.wso[0] >= 0 || A.wso[1] >= 0 ? nreg : 1;
        for (int tl = wp; tl < nsr * 8; tl += FL_T / 32) {   /* (warp-uniform: every lane takes the same tiles) */
            const int r = tl >> 3, j0 = (tl & 7) * 8, nsl = r ? 2 : A.nsl;
            if (!nsl) continue;
            const int jb = j0 + g, bz = (org[r][0] >> 1) + (jb >> 4), by = (org[r][1] >> 1) + (jb >> 2 & 3), bx = (org[r][2] >> 1) + (jb & 3);
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            for (int s_ = 0; s_ < 4; s_++) {
                unsigned bf[2];
                for (int u = 0; u < 2; u++) {   /* taps 16 s + 8 u + 2 tq, + 1: row 4 s + 2 u + tq / 2, x offset 2 (tq & 1) */
                    const int rr = 4 * s_ + 2 * u + tq / 2, z = 2 * bz - 1 + rr / 4, y = 2 * by - 1 + rr % 4, x = 2 * bx - 1 + 2 * (tq & 1);
                    const __half2 p2 = __halves2half2(tap(z, y, x), tap(z, y, x + 1)); bf[u] = *(const unsigned *)&p2;
                }
                mma16816<f16>(c4, af[s_], bf);
            }
            for (int r4 = 0; r4 < 4; r4++) { const int c = g + (r4 >> 1) * 8, b = 2 * tq + (r4 & 1); if (c < C) { const float v = c4[r4] + b1[c]; U.shw[wp][b][c] = v / (1.f + __expf(-v)); } }
            __syncwarp();
            const int b = lane & 7, j = j0 + b;
            for (int k = lane >> 3; k < nsl; k += 4) {
                const int o = r ? A.wso[k] : A.so[k];
                if (o < 0) continue;
                float v = 0.f; for (int c = 0; c < C; c++) v += W2[o * C + c] * U.shw[wp][b][c];
                sst[r][j][k] = v;
            }
            __syncwarp();
        }
    }
    __syncthreads();
    /* the upsampling, lat_up4_k's chain per (channel, latent cell lc, z offset rz): 16 voxels each into the planes L (fp16) */
    {
        const int ni = (A.nch + (nreg - 1) * 2) * 8 * 4;
        for (int it = tid; it < ni; it += FL_T) {
            const int rz = it & 3, lc = it >> 2 & 7, kk = it >> 5, cz = lc >> 2, cy = lc >> 1 & 1, cx = lc & 1;
            const float *S = kk < A.nch ? cs0[kk] : csw[(kk - A.nch) >> 1][(kk - A.nch) & 1];
            __half *Lo = kk < A.nch ? U.L.l0[kk] : U.L.lw[(kk - A.nch) >> 1][(kk - A.nch) & 1];
            float p[3][3];
#pragma unroll
            for (int b = 0; b < 3; b++)
#pragma unroll
                for (int e = 0; e < 3; e++) { const int o = (cy + b) * 4 + cx + e; p[b][e] = fl_lw(rz, S[cz * 16 + o], S[(cz + 1) * 16 + o], S[(cz + 2) * 16 + o]); }
#pragma unroll
            for (int ry = 0; ry < 4; ry++) {
                float q[3];
#pragma unroll
                for (int e = 0; e < 3; e++) q[e] = fl_lw(ry, p[0][e], p[1][e], p[2][e]);
#pragma unroll
                for (int rx = 0; rx < 4; rx++) Lo[(4 * cz + rz) * 64 + (4 * cy + ry) * 8 + 4 * cx + rx] = __float2half_rn(fl_lw(rx, q[0], q[1], q[2]));
            }
        }
    }
    __syncthreads();
    const int lz = tid >> 6, ly = (tid >> 3) & 7, lx = tid & 7, blk = ((lz >> 1) * 4 + (ly >> 1)) * 4 + (lx >> 1), sub = 16 * (lz & 3) + 4 * (ly & 3) + (lx & 3);
    /* the voxel's logit of staged channel k in region r (regions 1-3: the winding pair, k = ndm, ndm + 1) */
    auto logit = [&](int r, int k) -> float {
        float a = __half2float(r ? U.L.lw[r - 1][k - A.ndm][tid] : U.L.l0[k][tid]);
        const int c = A.ch[k];
        if (!r && A.sbase >= 0 && c < 2)
            a = fl_h(a + ldv(lg, (size_t)(A.sbase + 64 * c + sub) * cv + ((size_t)((org[0][0] + lz) >> 2) * A.h + ((org[0][1] + ly) >> 2)) * A.w + ((org[0][2] + lx) >> 2)));
        const int sl = r ? (A.wso[k - A.ndm] >= 0 ? k - A.ndm : -1) : A.slot[k];
        if (C && sl >= 0) a = fl_h(a + sst[r][blk][sl]);
        return a;
    };
    auto mat = [&](size_t i) { return bits ? (bits[i >> 5] >> (i & 31)) & 1u : (unsigned)((ct ? ct : ctx)[i] != 0); };
    const int gz = fz * 8 + lz, gy = fy * 8 + ly, gx = fx * 8 + lx, wz = gz - A.tz, wy = gy - A.ty, wx = gx - A.tx;
    const bool in = !(gz >= A.bz || gy >= A.by || gx >= A.bx || wz < A.hz || wz >= A.D - A.hz || wy < A.halo || wy >= A.W - A.halo || wx < A.halo || wx >= A.W - A.halo);
    const size_t i = in ? ((size_t)wz * A.H + wy) * A.W + wx : 0;
    const bool mi = in && mat(i);
    if (A.dbg && in) for (int k = 0; k < A.nch; k++) A.dbg[(size_t)k * A.D * hw + i] = logit(0, k);
    float v[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, rmax = 0.f, t9[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (in) {   /* fit_accum_k's per-voxel values */
        v[0] += 1.f;
        if (mi) {
            auto sg = [&](int k) { return 1.f / (1.f + __expf(-logit(0, k))); };
            v[1] += 1.f; v[2] += 2.f * sg(0) - 1.f; v[3] += 2.f * sg(1) - 1.f;
            for (int k = 0; k < 6; k++) v[4 + k] += sg(2 + k) - (k < 3 ? 0.f : 0.5f);
            { const float sd = sg(8); v[10] += (A.dcode ? fminf(sd, 0.999f) / (1.f - fminf(sd, 0.999f)) : sd) / A.dscale; }
            if (A.ch_recto >= 0) rmax = fmaxf(rmax, sg(9));
        }
    }
    if (A.ch_w4 >= 0 && mi) {   /* fit_accum_w4_k's terms (0 for a skipped one: its sums start at +0 and never become -0) */
        const int kw = A.ndm;
        const float pr = 2.f / (1.f + __expf(-logit(0, kw))) - 1.f, pi = 2.f / (1.f + __expf(-logit(0, kw + 1))) - 1.f;
        for (int a = 0; a < 3; a++) {
            if (!mat(i + (a == 0 ? A.dd * hw : a == 1 ? (size_t)A.dd * A.W : (size_t)A.dd))) continue;
            const float qr = 2.f / (1.f + __expf(-logit(1 + a, kw))) - 1.f, qi = 2.f / (1.f + __expf(-logit(1 + a, kw + 1))) - 1.f;
            t9[3 * a] = qr * pr + qi * pi; t9[3 * a + 1] = qi * pr - qr * pi; t9[3 * a + 2] = 1.f;
        }
    }
    __syncthreads();   /* (the planes L are read: red overlays them) */
    for (int k = 0; k < 11; k++) U.red[k][tid] = v[k];
    U.red[11][tid] = rmax;
    __syncthreads();
    for (int o = 256; o > 0; o >>= 1) {   /* fit_accum_k's tree */
        if (tid < o) { for (int k = 0; k < 11; k++) U.red[k][tid] += U.red[k][tid + o]; U.red[11][tid] = fmaxf(U.red[11][tid], U.red[11][tid + o]); }
        __syncthreads();
    }
    const size_t base = ((size_t)fz * A.F1 + fy) * A.F2 + fx;
    if (tid == 0 && U.red[0][0] > 0.f) {
        for (int k = 0; k < 11; k++) if (U.red[k][0] != 0.f) atomicAdd(&acc[k * A.nf + base], U.red[k][0]);
        atomicMax((int *)&acc[11 * A.nf + base], __float_as_int(U.red[11][0]));
    }
    if (A.ch_w4 < 0) return;
    __syncthreads();
    for (int k = 0; k < 9; k++) U.red[k][tid] = t9[k];
    __syncthreads();
    if (tid < 9) {   /* fit_accum_w4_k's serial sums */
        float s = 0.f;
#pragma unroll 16
        for (int t = 0; t < FL_T; t++) s += U.red[tid][t];
        if (s != 0.f) atomicAdd(&acc4[tid * A.nf + base], s);
    }
}
extern "C" int nn_fit_accum_lat(const void *lg, int lh16, int d, int h, int w, int sbase, const float *stem, int C, const int *so, const uint8_t *ctx, const uint8_t *ct,
                                const uint32_t *bits, float mean, float isd, int hz, int halo, int tz, int ty, int tx, const int box[3], int fh, const int F[3],
                                int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, int dcode, float *acc, int ch_w4, int dd, float *acc4) {
    if (fh != 8 || h != w || (stem && (C < 1 || C > FL_C || !ctx)) || (!bits && !ct && !ctx)) return 1;
    if ((tz & 3) || (ty & 3) || (tx & 3) || (ch_w4 >= 0 && (dd & 3))) return 1;   /* (field cells and the partner regions on the cell grid) */
    fl_args A = {};
    A.d = d; A.h = h; A.w = w; A.D = 4 * d; A.H = 4 * h; A.W = 4 * w; A.hz = hz; A.halo = halo; A.tz = tz; A.ty = ty; A.tx = tx; A.bz = box[0]; A.by = box[1]; A.bx = box[2];
    A.ch_recto = ch_recto; A.dcode = dcode; A.dscale = dscale; A.ch_w4 = ch_w4; A.dd = dd; A.sbase = sbase; A.C = stem ? C : 0; A.mean = mean; A.isd = isd; A.dbg = g_fl_dbg;
    if (ch_w4 >= 0 && (dd > halo || dd > hz || dd < 1)) { fprintf(stderr, "nn_fit_accum_lat: offset %d outside the halo (%d, z %d)\n", dd, halo, hz); abort(); }
    int n = 0;
    A.ch[n++] = ch_phase; A.ch[n++] = ch_phase + 1; for (int k = 0; k < 6; k++) A.ch[n++] = ch_normal + k; A.ch[n++] = ch_density; if (ch_recto >= 0) A.ch[n++] = ch_recto;
    A.ndm = n; if (ch_w4 >= 0) { A.ch[n++] = ch_w4; A.ch[n++] = ch_w4 + 1; }
    A.nch = n; A.nsl = 0; A.wso[0] = A.wso[1] = -1;
    for (int k = 0; k < n; k++) {
        const int o = A.C && A.ch[k] < 16 ? so[A.ch[k]] : -1;
        A.slot[k] = -1;
        if (o < 0) continue;
        if (A.nsl >= FL_MAXS) return 1;
        A.slot[k] = A.nsl; A.so[A.nsl++] = o;
        if (k >= A.ndm) A.wso[k - A.ndm] = o;
    }
    int f0[3], f1[3];
    const int t[3] = {tz, ty, tx}, wn[3] = {A.D, A.W, A.W}, hn[3] = {hz, halo, halo};
    for (int k = 0; k < 3; k++) {
        int lo = t[k] + hn[k], hi = t[k] + wn[k] - hn[k]; if (lo < 0) lo = 0; if (hi > box[k]) hi = box[k];
        f0[k] = lo / fh; f1[k] = (hi + fh - 1) / fh; if (f1[k] > F[k]) f1[k] = F[k];
        if (f1[k] <= f0[k]) return 0;
    }
    A.fz0 = f0[0]; A.fy0 = f0[1]; A.fx0 = f0[2]; A.nfy = f1[1] - f0[1]; A.nfx = f1[2] - f0[2]; A.F1 = F[1]; A.F2 = F[2];
    A.nf = (size_t)F[0] * F[1] * F[2];
    const size_t nb = (size_t)(f1[0] - f0[0]) * A.nfy * A.nfx;
    if (nb > 0x7fffffff || A.nf > 0x7fffffff) return 1;
    if (lh16) fit_accum_lat_k<f16><<<(unsigned)nb, FL_T>>>((const f16 *)lg, stem, ctx, ct, bits, A, acc, acc4);
    else fit_accum_lat_k<float><<<(unsigned)nb, FL_T>>>((const float *)lg, stem, ctx, ct, bits, A, acc, acc4);
    KCHECK();
    return 0;
}

/* fit.c fit_pool (no phase gradient, no CT spacing) of the device fields: one thread per pooled cell, the same sums in the same order
   in double with the roundings of the host (_rn: no fused multiply-adds), so the planes are byte-identical to the host's. dfield: the
   10 planes of n = D H W cells back to back (pc ps t0..t5 sp w), h the field cell size; out: 10 device planes of the pooled box; *cnt +=
   its cells with material. */
__global__ void fit_pool_k(const uint8_t *__restrict__ dfield, int D, int H, int W, double h, int fp, int pD, int pH, int pW, uint8_t *__restrict__ out, unsigned long long *cnt_) {
    const size_t np = (size_t)pD * pH * pW, o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    uint8_t v[10] = {128, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    if (o < np) {
        const size_t n = (size_t)D * H * W; const int x = (int)(o % pW), y = (int)((o / pW) % pH), z = (int)(o / ((size_t)pW * pH));
        const uint8_t *w = dfield + 9 * n, *sp = dfield + 8 * n;
        double sw = 0, st[6] = {0, 0, 0, 0, 0, 0}, srho = 0, swr = 0; int cnt = 0;
        for (int a = 0; a < fp && z * fp + a < D; a++) for (int b = 0; b < fp && y * fp + b < H; b++) for (int e = 0; e < fp && x * fp + e < W; e++) {
            const size_t i = ((size_t)(z * fp + a) * H + y * fp + b) * W + x * fp + e; cnt++;
            if (!w[i]) continue;
            const double wi = __ddiv_rn((double)w[i], 255.0); sw = __dadd_rn(sw, wi);
            for (int k = 0; k < 6; k++) st[k] = __dadd_rn(st[k], __dmul_rn(wi, (double)dfield[(size_t)(2 + k) * n + i]));
            if (sp[i]) { srho = __dadd_rn(srho, __ddiv_rn(__dmul_rn(wi, 8.0), __dmul_rn(h, (double)sp[i]))); swr = __dadd_rn(swr, wi); }
        }
        if (sw > 0) {
            v[9] = (uint8_t)llround(__ddiv_rn(__dmul_rn(255.0, sw), (double)cnt));
            for (int k = 0; k < 6; k++) v[2 + k] = (uint8_t)llround(__ddiv_rn(st[k], sw));
            if (swr > 0) { const double r = __ddiv_rn(8.0, __dmul_rn(__dmul_rn(h, (double)fp), __ddiv_rn(srho, swr))); v[8] = (uint8_t)(r < 1 ? 1 : r > 255 ? 255 : llround(r)); }
        }
        for (int k = 0; k < 10; k++) out[(size_t)k * np + o] = v[k];
    }
    const unsigned m = __ballot_sync(0xffffffffu, v[9] != 0);
    if ((threadIdx.x & 31) == 0 && m) atomicAdd(cnt_, (unsigned long long)__popc(m));
}
extern "C" void nn_fit_pool(const uint8_t *dfield, int D, int H, int W, double h, int fp, uint8_t *out, unsigned long long *cnt) {
    const int pD = (D + fp - 1) / fp, pH = (H + fp - 1) / fp, pW = (W + fp - 1) / fp; const size_t np = (size_t)pD * pH * pW;
    fit_pool_k<<<nblk(np, 256), 256>>>(dfield, D, H, W, h, fp, pD, pH, pW, out, cnt); KCHECK();
}

/* analytic data terms (nn_fit_loss2): one evaluation per voxel (src/fit_terms.h fit_vox2) */
__global__ void fit_loss2_k(fit_dev f, long long *pi, long long *gi) {   /* (fixed-point sums: deterministic) */
    __shared__ long long red[FIT_NPARTS][FIT_TPB];
    double lp[FIT_NPARTS] = {0}; long long lq[FIT_NPARTS] = {0};
    const size_t n = (size_t)f.D * f.H * f.W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const float wv = i < n && (f.stride <= 1 || fit_hash((unsigned)fit_box_index(&f, i) ^ f.salt) % (unsigned)f.stride == 0) ? f.w[i] * (1.f / 255.f) : 0.f;
    if (wv > 0.f) {
        size_t ic[FIT_VOX2_MAXC]; float gv[FIT_VOX2_MAXC];
        const int nc = fit_vox2(&f, i, wv, (float)(f.stride > 1 ? f.stride : 1), lp, ic, gv);
        for (int c = 0; c < nc; c++) if (gv[c] != 0.f) fit_gadd(gi, ic[c], gv[c]);
        for (int q = 0; q < FIT_NPARTS; q++) lq[q] = fit_pfix(lp[q]);
    }
    fit_reduce_parts_fix(lq, red, pi);
}
__global__ void fit_loss2_list_k(fit_dev f, const unsigned *list, const unsigned *count, unsigned cap, long long *pi, long long *gi) {
    __shared__ long long red[FIT_NPARTS][FIT_TPB];
    long long lp[FIT_NPARTS] = {0};   /* (per voxel in fixed point: the list order does not matter) */
    const unsigned nc = min(*count, cap);
    const float sc = (float)f.stride;
    for (unsigned k = blockIdx.x * blockDim.x + threadIdx.x; k < nc; k += gridDim.x * blockDim.x) {
        const size_t i = list[k];
        const float wv = f.w[i] * (1.f / 255.f);
        double vp[FIT_NPARTS] = {0};
        size_t ic[FIT_VOX2_MAXC]; float gv[FIT_VOX2_MAXC];
        const int ncr = fit_vox2(&f, i, wv, sc, vp, ic, gv);
        for (int c = 0; c < ncr; c++) if (gv[c] != 0.f) fit_gadd(gi, ic[c], gv[c]);
        for (int q = 0; q < FIT_NPARTS; q++) lp[q] += fit_pfix(vp[q]);
    }
    fit_reduce_parts_fix(lp, red, pi);
}
/* Minibatch steps (stride > 1): the step's voxels (hash(i ^ salt) % stride == 0, material) are first listed (warp-aggregated appends),
   then evaluated by a grid-stride kernel over the list (fit_loss2_list_k; fit_loss2_k's thread per voxel of the slab was 60-100 ms a
   step on a 1024^2 x 256 slab). The list from the other side: fit_hash is a bijection of 32-bit words, so {i < n : hash(i ^ salt) %
   stride == 0} is {unhash(k stride) ^ salt : k stride < 2^32} restricted to i < n: one thread per multiple of the stride (2^32 / stride,
   ~7.6 M at stride 562) reading the weight byte of its candidate. Hashing every voxel of the box instead took 3.7 ms a step on a
   96 x 3328 x 3520 slab (half of a fit's GPU time); fit_select_k still does where the box has fewer voxels than candidates. The list
   order varies run to run; the sums are fixed point (order-free).
   n < 2^32 (nn_fit_loss2's guard). */
__global__ void fit_select_k(fit_dev f, unsigned *list, unsigned *count, unsigned cap) {   /* the same list, one thread per voxel */
    const size_t n = (size_t)f.D * f.H * f.W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const bool sel = i < n && fit_hash((unsigned)fit_box_index(&f, i) ^ f.salt) % (unsigned)f.stride == 0 && f.w[i] != 0;
    const unsigned m = __ballot_sync(0xffffffffu, sel);
    if (!m) return;
    const int lane = threadIdx.x & 31, leader = __ffs(m) - 1;
    unsigned base = 0;
    if (lane == leader) base = atomicAdd(count, (unsigned)__popc(m));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (sel) { const unsigned k = base + __popc(m & ((1u << lane) - 1u)); if (k < cap) list[k] = (unsigned)i; }
}
__global__ void fit_select_inv_k(fit_dev f, unsigned long long nk, unsigned *list, unsigned *count, unsigned cap) {
    const size_t n = fit_box_cells(&f); const unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned i = 0; bool sel = false;
    if (k < nk) { const unsigned u = fit_unhash((unsigned)(k * (unsigned long long)(unsigned)f.stride)) ^ f.salt; size_t j = 0;   /* (u: a box cell; i: its array voxel) */
                  sel = (size_t)u < n && fit_crop_index(&f, u, &j) && f.w[j] != 0; i = (unsigned)j; }
    const unsigned m = __ballot_sync(0xffffffffu, sel);
    if (!m) return;
    const int lane = threadIdx.x & 31, leader = __ffs(m) - 1;
    unsigned base = 0;
    if (lane == leader) base = atomicAdd(count, (unsigned)__popc(m));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (sel) { const unsigned q = base + __popc(m & ((1u << lane) - 1u)); if (q < cap) list[q] = i; }
}
static unsigned *g_sel_list, *g_sel_count; static unsigned g_sel_cap;
extern "C" void nn_fit_loss2(const fit_dev *f, double *parts, float *g_delta) {
    const size_t n = fit_box_cells(f), na = (size_t)f->D * f->H * f->W, nl = (size_t)f->lz * f->ly * f->lx;   /* (the box's cells; na: the arrays') */
    long long *gi = fit_fix_scratch(nl), *pi = g_fix_p;
    if (f->stride <= 1 || n >= 0xffffffffull) {
        fit_loss2_k<<<nblk(na, FIT_TPB), FIT_TPB>>>(*f, pi, gi);
        fit_gconv_k<<<nblk(nl, 256) < 4096 ? nblk(nl, 256) : 4096, 256>>>(gi, nl, g_delta); fit_pconv_k<<<1, 32>>>(pi, parts); KCHECK(); return;
    }
    /* capacity: the expected n / stride plus a quarter (the count is binomial: sd ~ sqrt(n / stride), ~0.07% of it at 2 M) */
    const size_t want = n / (size_t)f->stride + n / (size_t)f->stride / 4 + 65536;
    if (want > g_sel_cap) {
        if (g_sel_list) CK(cudaFree(g_sel_list));
        if (!g_sel_count) CK(cudaMalloc(&g_sel_count, sizeof(unsigned)));
        CK(cudaMalloc(&g_sel_list, want * sizeof(unsigned))); g_sel_cap = (unsigned)want;
    }
    CK(cudaMemsetAsync(g_sel_count, 0, sizeof(unsigned)));
    const unsigned long long nk = ((1ull << 32) + (unsigned)f->stride - 1) / (unsigned)f->stride;   /* the multiples of the stride below 2^32 */
    if (nk < n) fit_select_inv_k<<<(unsigned)((nk + 255) / 256), 256>>>(*f, nk, g_sel_list, g_sel_count, g_sel_cap);
    else fit_select_k<<<nblk(na, 256), 256>>>(*f, g_sel_list, g_sel_count, g_sel_cap);   /* (small boxes: fewer voxels than candidates) */
    const unsigned nb = nblk(want, FIT_TPB) < 1024 ? nblk(want, FIT_TPB) : 1024;
    fit_loss2_list_k<<<nb, FIT_TPB>>>(*f, g_sel_list, g_sel_count, g_sel_cap, pi, gi);
    fit_gconv_k<<<nblk(nl, 256) < 4096 ? nblk(nl, 256) : 4096, 256>>>(gi, nl, g_delta); fit_pconv_k<<<1, 32>>>(pi, parts);
    KCHECK();
}

/* one grid point (row r, voxel (b, c)): the dense and the sparse grid kernels share it (the same float operations: the same bits) */
__device__ __forceinline__ float fit_grid_pt(const fit_dev *ms, const fit_grid_row &r, int b, int c, float y0, float x0, float step, int with_theta) {
    const float y = y0 + ((float)b + 0.5f) * step, x = x0 + ((float)c + 0.5f) * step, itp = 0.15915494309189535f;
    fit_pt p; fit_eval(ms[r.a], r.z - r.za, y, x, p);
    const float tha = p.th;
    float W = p.U - (float)ms[r.a].dir * p.th * itp;
    if (r.t > 0.f) { fit_pt q; fit_eval(ms[r.b], r.z - r.zb, y, x, q); W = (1.f - r.t) * W + r.t * (q.U - (float)ms[r.b].dir * q.th * itp); }
    return with_theta ? W + (float)ms[r.a].dir * tha * itp : W;
}
__global__ void fit_grid_k(const fit_dev *ms, const fit_grid_row *rows, int ny, int nx, float y0, float x0, float step, int with_theta, float *out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; const int a = blockIdx.y;
    if (i >= (size_t)ny * nx) return;
    const int b = (int)(i / nx), c = (int)(i % nx);
    out[(size_t)a * ny * nx + i] = fit_grid_pt(ms, rows[a], b, c, y0, x0, step, with_theta);
}
/* the grid's points whose mask bit is set (bit i of the row-major nz x ny x nx grid: word i / 64, bit i % 64), packed in index order:
   point i at out[woff[i / 64] + rank of bit i % 64 in its word] (woff: the exclusive prefix count of the set bits per word) */
__global__ void fit_grid_sparse_k(const fit_dev *ms, const fit_grid_row *rows, const unsigned long long *mask, const unsigned *woff, size_t nw, int ny, int nx,
                                  float y0, float x0, float step, int with_theta, float *out) {
    const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nw) return;
    unsigned long long m = mask[w]; unsigned o = woff[w];
    const size_t per = (size_t)ny * nx;
    while (m) {
        const int j = __ffsll((long long)m) - 1; m &= m - 1;
        const size_t i = w * 64 + j; const int a = (int)(i / per); const size_t rr = i - (size_t)a * per;
        const int b = (int)(rr / nx), c = (int)(rr % nx);
        out[o++] = fit_grid_pt(ms, rows[a], b, c, y0, x0, step, with_theta);
    }
}
/* the label mask of a device code block on the GPU (fit --codes scoring): bit i = code 1..252 at voxel i, woff the exclusive prefix
   count of the set bits per word (integers: exactly the CPU's), the total and the counts before the voxels starts[0..nst) */
__global__ void fit_lmask_k(const uint8_t *c, size_t nv, size_t nw, unsigned long long *mask, unsigned *cnt) {
    const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nw) return;
    const size_t i0 = w * 64; unsigned long long b = 0;
    if (i0 + 64 <= nv) {
        const uint4 *q = (const uint4 *)(c + i0);
        for (int k = 0; k < 4; k++) {
            const uint4 v = q[k]; const unsigned u[4] = {v.x, v.y, v.z, v.w};
            for (int e = 0; e < 4; e++) for (int j = 0; j < 4; j++) { const unsigned char x = (unsigned char)(u[e] >> (8 * j)); b |= (unsigned long long)((unsigned char)(x - 1) < 252) << (16 * k + 4 * e + j); }
        }
    } else for (size_t j = 0; i0 + j < nv; j++) b |= (unsigned long long)((unsigned char)(c[i0 + j] - 1) < 252) << j;
    mask[w] = b; cnt[w] = (unsigned)__popcll(b);
}
__global__ void fit_scan1_k(unsigned *a, size_t n, unsigned *bsum) {   /* exclusive scan of each 1024-run in place; the run totals */
    __shared__ unsigned s[1024];
    const size_t i = (size_t)blockIdx.x * 1024 + threadIdx.x; const unsigned v = i < n ? a[i] : 0u;
    s[threadIdx.x] = v; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const unsigned t = threadIdx.x >= (unsigned)o ? s[threadIdx.x - o] : 0u; __syncthreads(); s[threadIdx.x] += t; __syncthreads(); }
    if (i < n) a[i] = s[threadIdx.x] - v;
    if (threadIdx.x == 1023) bsum[blockIdx.x] = s[1023];
}
__global__ void fit_scan2_k(unsigned *bsum, size_t nb, unsigned long long *tot) {   /* (one thread: the run totals' exclusive scan) */
    unsigned long long r = 0; for (size_t k = 0; k < nb; k++) { const unsigned t = bsum[k]; bsum[k] = (unsigned)r; r += t; } *tot = r;
}
__global__ void fit_scan3_k(unsigned *a, size_t n, const unsigned *bsum) {
    const size_t i = (size_t)blockIdx.x * 1024 + threadIdx.x; if (i < n) a[i] += bsum[blockIdx.x];
}
__global__ void fit_starts_k(const unsigned long long *mask, const unsigned *woff, const unsigned long long *st, int nst, unsigned long long *us) {
    const int k = threadIdx.x; if (k >= nst) return;
    const unsigned long long i0 = st[k]; const size_t w = i0 / 64; const int r = (int)(i0 % 64);
    us[k] = woff[w] + (r ? (unsigned long long)__popcll(mask[w] & ((1ull << r) - 1)) : 0ull);
}
/* tmp: >= (nw + 1023) / 1024 + 2 * 64 + 2 words of 8 bytes; nst <= 64 */
size_t nn_fit_label_mask(const uint8_t *dcodes, size_t nv, uint64_t *dmask, uint32_t *dwoff, void *dtmp, int nst, const uint64_t *starts, uint64_t *ustart) {
    const size_t nw = (nv + 63) / 64, nb = (nw + 1023) / 1024; if (!nw) return 0;
    unsigned long long *tot = (unsigned long long *)dtmp, *dst = tot + 1, *dus = dst + 64; unsigned *bsum = (unsigned *)(dus + 64);
    fit_lmask_k<<<(unsigned)((nw + 255) / 256), 256>>>(dcodes, nv, nw, (unsigned long long *)dmask, (unsigned *)dwoff);
    fit_scan1_k<<<(unsigned)nb, 1024>>>((unsigned *)dwoff, nw, bsum);
    fit_scan2_k<<<1, 1>>>(bsum, nb, tot);
    fit_scan3_k<<<(unsigned)nb, 1024>>>((unsigned *)dwoff, nw, bsum);
    if (nst > 0) { CK(cudaMemcpy(dst, starts, (size_t)nst * 8, cudaMemcpyHostToDevice));
                   fit_starts_k<<<1, 64>>>((const unsigned long long *)dmask, (const unsigned *)dwoff, dst, nst, dus); }
    CK(cudaGetLastError());
    unsigned long long t = 0; CK(cudaMemcpy(&t, tot, 8, cudaMemcpyDeviceToHost));
    if (nst > 0) CK(cudaMemcpy(ustart, dus, (size_t)nst * 8, cudaMemcpyDeviceToHost));
    return (size_t)t;
}
void nn_fit_grid_sparse(const fit_dev *ms, const fit_grid_row *rows, const uint64_t *mask, const uint32_t *woff, size_t nw, int ny, int nx, float y0, float x0, float step,
                        int with_theta, float *out) {
    if (!nw) return;
    fit_grid_sparse_k<<<(unsigned)((nw + 127) / 128), 128>>>(ms, rows, (const unsigned long long *)mask, (const unsigned *)woff, nw, ny, nx, y0, x0, step, with_theta, out);
    CK(cudaGetLastError());
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

/* fit_fields_load_stream's uploads (src/fit.h nn_fit_up_*): a thread copies pageable host ranges to the device through two pinned
   staging buffers (memcpy into one while the other's DMA runs, on its own stream), so the uploads run beside the main thread's
   decoding instead of after it (pageable cudaMemcpy: ~3 GB/s, synchronous). */
#include <pthread.h>
struct nn_fit_up {
    pthread_t th; pthread_mutex_t mu; pthread_cond_t cv; int dev, quit;
    struct job { uint8_t *dst; const uint8_t *src; size_t n; } *q; long nq, cap, done;   /* jobs [0, nq); [0, done) consumed */
    uint8_t *pin[2]; size_t S; cudaEvent_t ev[2]; cudaStream_t st;
};
static void *nn_fit_up_main(void *a) {
    nn_fit_up *u = (nn_fit_up *)a; CK(cudaSetDevice(u->dev)); long j = 0;
    for (;;) {
        pthread_mutex_lock(&u->mu);
        while (u->done == u->nq && !u->quit) pthread_cond_wait(&u->cv, &u->mu);
        if (u->done == u->nq) { pthread_mutex_unlock(&u->mu); break; }
        const nn_fit_up::job jb = u->q[u->done]; pthread_mutex_unlock(&u->mu);
        for (size_t o = 0; o < jb.n; o += u->S, j++) {
            const size_t len = jb.n - o < u->S ? jb.n - o : u->S; const int s = (int)(j & 1);
            CK(cudaEventSynchronize(u->ev[s]));
            memcpy(u->pin[s], jb.src + o, len);
            CK(cudaMemcpyAsync(jb.dst + o, u->pin[s], len, cudaMemcpyHostToDevice, u->st)); CK(cudaEventRecord(u->ev[s], u->st));
        }
        pthread_mutex_lock(&u->mu); u->done++; pthread_cond_broadcast(&u->cv); pthread_mutex_unlock(&u->mu);
    }
    return nullptr;
}
extern "C" nn_fit_up *nn_fit_up_open(void) {
    nn_fit_up *u = (nn_fit_up *)calloc(1, sizeof *u); u->S = (size_t)32 << 20;
    CK(cudaGetDevice(&u->dev)); pthread_mutex_init(&u->mu, nullptr); pthread_cond_init(&u->cv, nullptr);
    for (int s = 0; s < 2; s++) { CK(cudaMallocHost(&u->pin[s], u->S)); CK(cudaEventCreateWithFlags(&u->ev[s], cudaEventDisableTiming)); CK(cudaEventRecord(u->ev[s], 0)); }
    CK(cudaStreamCreateWithFlags(&u->st, cudaStreamNonBlocking));
    pthread_create(&u->th, nullptr, nn_fit_up_main, u);
    return u;
}
extern "C" long nn_fit_up_put(nn_fit_up *u, void *dst, const void *src, size_t n) {
    pthread_mutex_lock(&u->mu);
    if (u->nq == u->cap) { u->cap = u->cap ? 2 * u->cap : 64; u->q = (nn_fit_up::job *)realloc(u->q, u->cap * sizeof *u->q); }
    u->q[u->nq] = {(uint8_t *)dst, (const uint8_t *)src, n}; const long id = u->nq++;
    pthread_cond_broadcast(&u->cv); pthread_mutex_unlock(&u->mu);
    return id;
}
extern "C" void nn_fit_up_wait(nn_fit_up *u, long id) {   /* job id's source read (reusable) */
    pthread_mutex_lock(&u->mu); while (u->done <= id) pthread_cond_wait(&u->cv, &u->mu); pthread_mutex_unlock(&u->mu);
}
extern "C" void nn_fit_up_close(nn_fit_up *u) {   /* every job on the device; frees u */
    if (!u) return;
    pthread_mutex_lock(&u->mu); u->quit = 1; pthread_cond_broadcast(&u->cv); pthread_mutex_unlock(&u->mu);
    pthread_join(u->th, nullptr);
    CK(cudaStreamSynchronize(u->st)); CK(cudaStreamDestroy(u->st));
    for (int s = 0; s < 2; s++) { CK(cudaEventDestroy(u->ev[s])); CK(cudaFreeHost(u->pin[s])); }
    pthread_mutex_destroy(&u->mu); pthread_cond_destroy(&u->cv); free(u->q); free(u);
}
