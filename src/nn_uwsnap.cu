/* The slab fit's voxel unwrapping on the GPU (src/nn_uwsnap.h; src/fit.c uw_accum_snap, the same expressions line for line). A thread
   per lattice node (az, bb, cc) walks the field voxels nearest to it in (z, y, x) order: confidence (uw_q), wrapped U (uw_rec), the
   ref's integer (uw_refsnap) and the node's sums (uw_accum), in double. The fields are the fit's device planes (no upload); the tables,
   the two lattices (the ref's only over the planes the slab reads) and the byte-pair angle table are uploaded per call. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn_uwsnap.h"

struct uws_dev {
    int D, H, W, oy, ox; double h;
    const uint8_t *pc, *ps, *sp, *w;
    double min_conf, packed_um, um;
    const double *conf, *a2; const uws_z *zt; const uws_row *rm, *rr;
    int lz, ly, lx, L; double iL; int dir; const float *delta;
    int rL, rlx, rdir; double riL; const float *rdelta;
    double roff1, roff2, rbias; int32_t kb;
    const int *zr, *yr, *xr;
};

/* fit.c ev_party_W */
__device__ __forceinline__ double uws_W(const uws_row &q, const float *delta, int L, int lx, double iL, double pit, double off, int dir, double x, double r, double t2) {   /* t2: theta / 2 pi */
    const double qx = iL > 0 ? x * iL : x / L, ax = qx < 0 ? 0 : qx > lx - 1 ? lx - 1 : qx;
    int c0x = (int)ax; if (c0x > lx - 2) c0x = lx - 2 < 0 ? 0 : lx - 2;
    const double ctx = lx > 1 ? ax - c0x : 0, wx[2] = {1 - ctx, ctx};
    const size_t xo[2] = {(size_t)(c0x < lx ? c0x : lx - 1), (size_t)(c0x + 1 < lx ? c0x + 1 : lx - 1)};
    double s = 0;
    for (int c = 0; c < 8; c++) { const double w = q.wzy[c >> 1] * wx[c & 1]; if (w != 0) s += w * delta[q.base[c >> 1] + xo[c & 1]]; }
    const double U = r / pit + off + s;
    return U - dir * t2;   /* (dir +-1: dir * th / (2 pi) == dir * (th / (2 pi)) bit for bit) */
}

/* one voxel of uw_accum_snap: its term added to *a / *ws (a row's running sums) */
__device__ __forceinline__ void uws_vox(const uws_dev &P, const uws_z &Z, const uws_row &Q, const uws_row &QR, int z, int y, int xx, double *a, double *ws,
                                        unsigned *nr, unsigned *over) {
    const size_t i = z * (size_t)P.H * P.W + (size_t)y * P.W + xx;
    /* uw_q (conf and ang / 2 pi from per-byte-pair tables of the host's own expressions) */
    if (!P.w[i]) return;
    const int pp = P.pc[i] << 8 | P.ps[i]; const double conf = P.conf[pp];
    if (conf < P.min_conf) return;
    if (P.packed_um > 0 && P.sp[i] && P.sp[i] * P.h / 8.0 * (P.um > 0 ? P.um : 2.4) < 0.6 * P.packed_um) return;
    const float q = (float)(conf * P.w[i] / 255.0);
    if (q <= 0) return;
    /* uw_rec */
    const double th = atan2((y + P.oy + 0.5) * P.h - Z.cy, (xx + P.ox + 0.5) * P.h - Z.cx), t2 = th / (2 * M_PI);
    double u = P.a2[pp] + P.dir * t2; u -= floor(u);
    const float uw = (float)u;
    /* uw_refsnap + uw_accum */
    const double py = (y + P.oy + 0.5) * P.h, px = (xx + P.ox + 0.5) * P.h, dy = py - Z.cy, dx = px - Z.cx, yr = py + P.roff1, xr = px + P.roff2;
    const double rr = hypot(dy, dx);
    const double Wr = Z.same ? uws_W(QR, P.rdelta, P.rL, P.rlx, P.riL, Z.rpit, Z.roff, P.rdir, xr, rr, t2)
                             : uws_W(QR, P.rdelta, P.rL, P.rlx, P.riL, Z.rpit, Z.roff, P.rdir, xr, hypot(yr - Z.gy, xr - Z.gx), atan2(yr - Z.gy, xr - Z.gx) / (2 * M_PI));
    const int32_t kr = (int32_t)round(Wr + P.dir * t2 - P.rbias - uw) - P.kb;
    if (!(kr > 0 && kr <= INT16_MAX)) { (*over)++; return; }
    (*nr)++;
    const double Um = uws_W(Q, P.delta, P.L, P.lx, P.iL, Z.pit, Z.off, P.dir, px, rr, t2) + P.dir * t2, Uv = (double)uw + (P.kb + (int16_t)kr);
    *a += q * (Uv - Um); *ws += q;
}

/* a thread per node: its voxels in the CPU's (z, y, x) order, so its sums are the CPU's up to atan2 / hypot. FP64-bound: a block per
   node with a thread per field row (row sums added in order) was no faster (1667 s030, desk 5060 Ti: 504 / 404 vs 524 / 337 ms) */
__global__ void k_uw_snap(uws_dev P, double *acc, double *wsum, unsigned long long *cnt) {
    const size_t nl = (size_t)P.lz * P.ly * P.lx;
    for (size_t kn = blockIdx.x * (size_t)blockDim.x + threadIdx.x; kn < nl; kn += (size_t)gridDim.x * blockDim.x) {
        const int az = (int)(kn / ((size_t)P.ly * P.lx)), bb = (int)((kn / P.lx) % P.ly), cc = (int)(kn % P.lx);
        double a = 0, ws = 0; unsigned nr = 0, over = 0;
        const int x0 = P.xr[cc], x1 = P.xr[cc + 1];
        if (x0 < x1)
            for (int z = P.zr[az]; z < P.zr[az + 1]; z++)
                for (int y = P.yr[bb]; y < P.yr[bb + 1]; y++) {
                    const uws_row Q = P.rm[(size_t)z * P.H + y], QR = P.rr[(size_t)z * P.H + y];
                    for (int xx = x0; xx < x1; xx++) uws_vox(P, P.zt[z], Q, QR, z, y, xx, &a, &ws, &nr, &over);
                }
        acc[kn] = a; wsum[kn] = ws;
        if (nr) atomicAdd(&cnt[0], (unsigned long long)nr);
        if (over) atomicAdd(&cnt[1], (unsigned long long)over);
    }
}
#define UWS_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "uw snap gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); rc = -1; goto done; } } while (0)

extern "C" int nn_uw_snap(const uws_args *a, double *acc, double *wsum, size_t *reached, size_t *over) {
    const size_t nl = (size_t)a->lz * a->ly * a->lx, nrow = (size_t)a->D * a->H;
    const size_t s_ang = 2 * 65536 * sizeof(double), s_zt = (size_t)a->D * sizeof(uws_z), s_row = nrow * sizeof(uws_row), s_d = nl * sizeof(float),
                 s_rd = a->rn * sizeof(float), s_idx = ((size_t)a->lz + 1 + a->ly + 1 + a->lx + 1) * sizeof(int), s_out = 2 * nl * sizeof(double);
    const size_t tot = s_ang + s_zt + 2 * s_row + s_d + s_rd + s_idx + s_out + 16 + 8 * 16;
    char *buf = nullptr; int rc = 0; unsigned long long hc[2] = {0, 0};
    uws_dev P;
    if (cudaMalloc(&buf, tot) != cudaSuccess) { cudaGetLastError(); return -1; }
    {
        size_t o = 0; auto take = [&](size_t s) { char *p = buf + o; o += (s + 15) / 16 * 16; return p; };
        double *dang = (double *)take(s_ang); uws_z *dzt = (uws_z *)take(s_zt); uws_row *drm = (uws_row *)take(s_row), *drr = (uws_row *)take(s_row);
        float *dd = (float *)take(s_d), *drd = (float *)take(s_rd); int *didx = (int *)take(s_idx); double *dout = (double *)take(s_out);
        unsigned long long *dcnt = (unsigned long long *)take(16);
        UWS_CK(cudaMemcpy(dang, a->tab, s_ang, cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(dzt, a->zt, s_zt, cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(drm, a->rm, s_row, cudaMemcpyHostToDevice)); UWS_CK(cudaMemcpy(drr, a->rr, s_row, cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(dd, a->delta, s_d, cudaMemcpyHostToDevice)); if (s_rd) UWS_CK(cudaMemcpy(drd, a->rdelta, s_rd, cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(didx, a->zr, (a->lz + 1) * sizeof(int), cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(didx + a->lz + 1, a->yr, (a->ly + 1) * sizeof(int), cudaMemcpyHostToDevice));
        UWS_CK(cudaMemcpy(didx + a->lz + 1 + a->ly + 1, a->xr, (a->lx + 1) * sizeof(int), cudaMemcpyHostToDevice));
        UWS_CK(cudaMemset(dcnt, 0, 16));
        P = {a->D, a->H, a->W, a->oy, a->ox, a->h, a->pc, a->ps, a->sp, a->w, a->min_conf, a->packed_um, a->um, dang, dang + 65536, dzt, drm, drr,
             a->lz, a->ly, a->lx, a->L, a->iL, a->dir, dd, a->rL, a->rlx, a->rdir, a->riL, drd, a->roff1, a->roff2, a->rbias, a->kb,
             didx, didx + a->lz + 1, didx + a->lz + 1 + a->ly + 1};
        const int TB = 128; const unsigned nb = (unsigned)((nl + TB - 1) / TB);
        cudaEvent_t e0, e1; const int tm = getenv("UFSM_UW_GPU_TIMES") != nullptr; if (tm) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); }
        k_uw_snap<<<nb, TB>>>(P, dout, dout + nl, dcnt);
        UWS_CK(cudaGetLastError());
        if (tm) {   /* UFSM_UW_GPU_TIMES: the kernel 4 more times (same outputs: it overwrites them; counters reset), the fastest reported */
            cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0, best = 0; cudaEventElapsedTime(&best, e0, e1);
            for (int r = 0; r < 4; r++) { cudaMemset(dcnt, 0, 16); cudaEventRecord(e0); k_uw_snap<<<nb, TB>>>(P, dout, dout + nl, dcnt); cudaEventRecord(e1); cudaEventSynchronize(e1);
                                          cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms; }
            fprintf(stderr, "uw gpu: kernel %.1f ms (best of 5)\n", best); cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        UWS_CK(cudaMemcpy(acc, dout, nl * sizeof(double), cudaMemcpyDeviceToHost)); UWS_CK(cudaMemcpy(wsum, dout + nl, nl * sizeof(double), cudaMemcpyDeviceToHost));
        UWS_CK(cudaMemcpy(hc, dcnt, 16, cudaMemcpyDeviceToHost));
    }
    *reached = (size_t)hc[0]; *over = (size_t)hc[1];
done:
    cudaFree(buf);
    return rc;
}
