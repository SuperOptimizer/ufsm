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
__global__ void fit_smooth_k(const float *v, const float *lap, const float *wn, int lz, int ly, int lx, float ws, float wd, double *parts, float *g) {
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
    if (threadIdx.x == 0) atomicAdd(&parts[6], red[0]);
}
extern "C" void nn_fit_smooth(const float *delta, int lz, int ly, int lx, float w_smooth, float w_decay, const float *wn, float *lap, double *parts, float *g_delta) {
    const size_t n = (size_t)lz * ly * lx;
    fit_lap_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lz, ly, lx, lap);
    fit_smooth_k<<<nblk(n, FIT_TPB), FIT_TPB>>>(delta, lap, wn, lz, ly, lx, w_smooth, w_decay, parts, g_delta);
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

/* analytic data terms (nn_fit_loss2): one evaluation per voxel (src/fit_terms.h fit_vox2) */
__global__ void fit_loss2_k(fit_dev f, double *parts, float *g_delta) {
    __shared__ double red[FIT_NPARTS][FIT_TPB];
    double lp[FIT_NPARTS] = {0};
    const size_t n = (size_t)f.D * f.H * f.W, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const float wv = i < n && (f.stride <= 1 || fit_hash((unsigned)i ^ f.salt) % (unsigned)f.stride == 0) ? f.w[i] * (1.f / 255.f) : 0.f;
    if (wv > 0.f) {
        size_t ic[FIT_VOX2_MAXC]; float gv[FIT_VOX2_MAXC];
        const int nc = fit_vox2(&f, i, wv, (float)(f.stride > 1 ? f.stride : 1), lp, ic, gv);
        for (int c = 0; c < nc; c++) if (gv[c] != 0.f) atomicAdd(&g_delta[ic[c]], gv[c]);
    }
    fit_reduce_parts(lp, red, parts);
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
