/* fit --phase-gate on the GPU (src/phasegate.c fit_phase_gate's per-cell pass, src/fit.h fit_phase_gate_gpu): on fields already on the
   device (the streamed fit's planes). Per block of field rows: (1) per cell the confidence-weighted radial phase rate and density
   radial rate (CPU: pg_rows pass 1; angle / length from the same float byte-pair tables), (2) box sums over +-R cells in x then y
   (direct float sums of 2R + 1 terms; the CPU runs double running sums: last-bit differences), (3) the gate and the phasor blend
   with W_ref (uploaded per block), written to the device pc / ps planes; statistics by atomics. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "phase gate gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); rc = -1; goto done; } } while (0)

static float *g_pgtab;   /* angle (turns) [65536], length x 127 [65536] at pc << 8 | ps */
static int pg_tab(void) {
    if (g_pgtab) return 0;
    float *h = (float *)malloc(2 * 65536 * sizeof(float)); if (!h) return -1;
    for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) { h[a << 8 | b] = (float)(atan2(b - 128.0, a - 128.0) / (2 * M_PI)); h[65536 + (a << 8 | b)] = (float)hypot(a - 128.0, b - 128.0); }
    const int bad = cudaMalloc(&g_pgtab, 2 * 65536 * sizeof(float)) != cudaSuccess || cudaMemcpy(g_pgtab, h, 2 * 65536 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess;
    free(h); if (bad) { cudaFree(g_pgtab); g_pgtab = nullptr; cudaGetLastError(); return -1; }
    return 0;
}

struct pg_dev { const uint8_t *pc, *ps, *t1, *t2, *t5, *sp, *w; int nz, H, W, R, mode; float h; const float *ay, *ax; const float *tab; };

__global__ void k_pg_rates(pg_dev P, float *pa, float *pb) {
    const size_t HW = (size_t)P.H * P.W, n = (size_t)P.nz * HW;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        pa[i] = pb[i] = 0;
        const int z = (int)(i / HW), y = (int)((i / P.W) % P.H), x = (int)(i % P.W);
        if (P.mode || !P.w[i] || y == 0 || x == 0 || y + 1 >= P.H || x + 1 >= P.W || !P.sp[i]) continue;
        const float *A = P.tab, *L = P.tab + 65536;
        const double c0 = L[P.pc[i] << 8 | P.ps[i]] / 127.0; if (c0 < 0.125) continue;
        double ry = (y + 0.5) * P.h - P.ay[z], rx = (x + 0.5) * P.h - P.ax[z]; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
        const double p0 = A[P.pc[i] << 8 | P.ps[i]];
        const size_t ny[2] = {i - P.W, i + P.W}, nx[2] = {i - 1, i + 1}; double gy = 0, gx = 0; int ok = 1;
        for (int s = 0; s < 2 && ok; s++) {
            const size_t a = ny[s], b = nx[s];
            if (L[P.pc[a] << 8 | P.ps[a]] < 16 || L[P.pc[b] << 8 | P.ps[b]] < 16) { ok = 0; break; }
            double dy = A[P.pc[a] << 8 | P.ps[a]] - p0, dx = A[P.pc[b] << 8 | P.ps[b]] - p0;
            dy -= floor(dy + 0.5); dx -= floor(dx + 0.5); gy += s ? dy : -dy; gx += s ? dx : -dx;
        }
        if (!ok) continue;
        const double rate = (0.5 * gy * ry + 0.5 * gx * rx) / P.h;
        const double nyy = P.t1[i] / 255.0, nxx = P.t2[i] / 255.0, nyx = P.t5[i] / 255.0 - 0.5;
        const double nr = sqrt(fmax(0.0, ry * ry * nyy + rx * rx * nxx + 2 * ry * rx * nyx)), dn = nr / (P.sp[i] * P.h / 8.0);
        pa[i] = (float)(c0 * rate); pb[i] = (float)(c0 * dn);
    }
}
__global__ void k_pg_boxx(pg_dev P, const float *a, const float *b, float *a2, float *b2) {   /* x box sums */
    const size_t n = (size_t)P.nz * P.H * P.W;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int x = (int)(i % P.W); const int x0 = x - P.R < 0 ? 0 : x - P.R, x1 = x + P.R >= P.W ? P.W - 1 : x + P.R;
        double s1 = 0, s2 = 0; for (int k = x0; k <= x1; k++) { s1 += a[i - x + k]; s2 += b[i - x + k]; }
        a2[i] = (float)s1; b2[i] = (float)s2;
    }
}
__global__ void k_pg_gate(pg_dev P, const float *a2, const float *b2, const float *wr, uint8_t *opc, uint8_t *ops, float lo, float hi, float rc,
                          unsigned long long *cnt, double *qs) {   /* y box sums, the gate, the blend; cnt: material, gated, n q */
    const size_t HW = (size_t)P.H * P.W, n = (size_t)P.nz * HW;
    unsigned long long lm = 0, lg = 0, lq = 0; double lqs = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint8_t c8 = P.pc[i], s8 = P.ps[i];
        if (!P.w[i]) { opc[i] = c8; ops[i] = s8; continue; }
        lm++;
        double g = 0;
        if (!P.mode) {
            const int y = (int)((i / P.W) % P.H); const size_t rb = i - (size_t)y * P.W;
            const int y0 = y - P.R < 0 ? 0 : y - P.R, y1 = y + P.R >= P.H ? P.H - 1 : y + P.R;
            double sa = 0, sb = 0; for (int k = y0; k <= y1; k++) { sa += a2[rb + (size_t)k * P.W]; sb += b2[rb + (size_t)k * P.W]; }
            if (sb > 1e-9) { const double q = sa / sb, t = q <= lo ? 0 : q >= hi ? 1 : (q - lo) / (hi - lo); g = t * t * (3 - 2 * t); lqs += q; lq++; }
        }
        const float w = wr[i];
        if (g < 1 && w == w) {
            float sr, cr; sincosf(6.2831853f * (w - floorf(w)), &sr, &cr);
            const double re = g * (c8 - 128.0) / 127.0 + (1 - g) * rc * cr, im = g * (s8 - 128.0) / 127.0 + (1 - g) * rc * sr;
            opc[i] = (uint8_t)llround(fmin(255.0, fmax(1.0, 128 + 127 * re))); ops[i] = (uint8_t)llround(fmin(255.0, fmax(1.0, 128 + 127 * im)));
            if (g < 0.5) lg++;
        } else { opc[i] = c8; ops[i] = s8; }
    }
    if (lm) atomicAdd(&cnt[0], lm); if (lg) atomicAdd(&cnt[1], lg); if (lq) { atomicAdd(&cnt[2], lq); atomicAdd(qs, lqs); }
}

/* one block of nz field rows: device planes (pc ps t1 t2 t5 sp w at the block's first row; pc / ps rewritten in place), host W_ref
   (nz x H x W), the axis per row (host, box-local native). st += {material, gated, sum q, n q}. Returns 0 / -1. */
extern "C" int fit_phase_gate_gpu(uint8_t *pc, uint8_t *ps, const uint8_t *t1, const uint8_t *t2, const uint8_t *t5, const uint8_t *sp, const uint8_t *w,
                                  int nz, int H, int W, double h, const float *ay, const float *ax, const float *wref, int mode, int R, double lo, double hi, double rconf, double st[4]) {
    int rc = 0;
    const size_t n = (size_t)nz * H * W;
    float *b4 = nullptr, *dw = nullptr, *dax = nullptr; uint8_t *o2 = nullptr; unsigned long long *cnt = nullptr; double *qs = nullptr;
    unsigned long long hc[3] = {0, 0, 0}; double hq = 0;
    pg_dev P = {pc, ps, t1, t2, t5, sp, w, nz, H, W, R, mode, (float)h, nullptr, nullptr, nullptr};
    const int TB = 256; int nb = (int)((n + TB - 1) / TB); if (nb > 65535 * 8) nb = 65535 * 8;
    if (pg_tab()) return -1;
    P.tab = g_pgtab;
    PG_CK(cudaMalloc(&b4, 4 * n * sizeof(float))); PG_CK(cudaMalloc(&dw, n * sizeof(float))); PG_CK(cudaMalloc(&o2, 2 * n));
    PG_CK(cudaMalloc(&dax, 2 * nz * sizeof(float))); PG_CK(cudaMalloc(&cnt, 3 * sizeof(unsigned long long))); PG_CK(cudaMalloc(&qs, sizeof(double)));
    PG_CK(cudaMemcpy(dw, wref, n * sizeof(float), cudaMemcpyHostToDevice));
    PG_CK(cudaMemcpy(dax, ay, nz * sizeof(float), cudaMemcpyHostToDevice)); PG_CK(cudaMemcpy(dax + nz, ax, nz * sizeof(float), cudaMemcpyHostToDevice));
    PG_CK(cudaMemset(cnt, 0, 3 * sizeof(unsigned long long))); PG_CK(cudaMemset(qs, 0, sizeof(double)));
    P.ay = dax; P.ax = dax + nz;
    k_pg_rates<<<nb, TB>>>(P, b4, b4 + n);
    k_pg_boxx<<<nb, TB>>>(P, b4, b4 + n, b4 + 2 * n, b4 + 3 * n);
    k_pg_gate<<<nb, TB>>>(P, b4 + 2 * n, b4 + 3 * n, dw, o2, o2 + n, (float)lo, (float)hi, (float)rconf, cnt, qs);
    PG_CK(cudaGetLastError());
    PG_CK(cudaMemcpy(pc, o2, n, cudaMemcpyDeviceToDevice)); PG_CK(cudaMemcpy(ps, o2 + n, n, cudaMemcpyDeviceToDevice));
    PG_CK(cudaMemcpy(hc, cnt, sizeof hc, cudaMemcpyDeviceToHost)); PG_CK(cudaMemcpy(&hq, qs, sizeof hq, cudaMemcpyDeviceToHost));
    st[0] += (double)hc[0]; st[1] += (double)hc[1]; st[2] += hq; st[3] += (double)hc[2];
done:
    cudaFree(b4); cudaFree(dw); cudaFree(o2); cudaFree(dax); cudaFree(cnt); cudaFree(qs);
    if (rc) cudaGetLastError();
    return rc;
}
