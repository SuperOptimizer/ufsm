/* ufsm merge-flag on the GPU (src/mfgpu.h, 2026-11-18). Tiles of ZS output rows x YS output columns (full x) with a 4-cell halo in z
   and y: per sum c of src/mf_core.h an x pass (the cell values computed from the bytes, summed over x +- r with x clamped), a y pass and
   a z pass (coordinates clamped to the box: edge replication as the CPU's mf_block), all int32 (exact); then mf_final per cell. The
   halo rows past the box edges are not stored: the clamped coordinate always lands on a stored row. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mf_core.h"
#include "mfgpu.h"

#define MG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "merge-flag gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)
enum { MG_B = 256, MG_ZS = 32, MG_HALO = 4 };

struct mg_fix;
struct mf_gpu {
    int dev; long long H, W, ys; uint8_t *pl[4], *out; int32_t *s, *ampq; mg_fix *fix; uint32_t *cnt, fcap; unsigned long long *cc; size_t cap_in, cap_out;
};
struct mg_geo { long long D, H, W, gzs, gys; int tz, ty; long long oz0, oy0; int nzo, nyo; };

__device__ __forceinline__ long long mg_cl(long long v, long long lo, long long hi) { return v < lo ? lo : v > hi ? hi : v; }

struct mg_pl { const uint8_t *p[4]; };
/* mf_final in fp32 (fp64 runs at 1/64 rate on these cards), deferred to mf_final itself (fp64, the CPU's arithmetic) where 255 p lands
   near a rounding edge: within 2 x 63.75 x a bound on fp32's error in the logit (atan2f / expf a few ulp: |dt| < 1.2e-7 per axis, so
   |dR5| < 5e-7 (sp5 + 1); amp9 / sp9 / the logit's sums ~1e-7 relative; the bound x ~3), + 1e-4. Returns -1 there (the cell goes to the fp64 list), else the byte. Material cells only. */
struct mg_fix { uint32_t k; int32_t s[MF_NS]; };
__device__ __forceinline__ int mg_final(const int32_t s[MF_NS], double b, double ca, double cr, double cs) {
    const float amp9 = s[1] > 0 ? (float)s[0] / ((float)MF_AMPQ * s[1]) : 0.f, sp9 = s[3] > 0 ? s[2] / (8.f * s[3]) : 0.f, sp5 = s[5] > 0 ? s[4] / (8.f * s[5]) : 0.f;
    float ka = 0;
#pragma unroll
    for (int a = 0; a < 3; a++) { const float t = atan2f((float)s[7 + 2 * a], (float)s[6 + 2 * a]) * (float)(0.5 / M_PI); ka += t * t; }
    const float lg = (float)b + (float)ca * amp9 + (float)cr * (sqrtf(ka) * sp5) + (float)cs * sp9, v = 255.f / (1.f + expf(-lg));
    const float tol = 127.5f * (fabsf((float)cr) * 1e-6f * (sp5 + 1.f) + (fabsf((float)ca) + fabsf((float)cs) * (sp9 + 1.f)) * 3e-7f + 3e-6f * (fabsf(lg) + 1.f)) + 1e-4f;
    const float fr = v - floorf(v);
    return fabsf(fr - 0.5f) < tol ? -1 : (int)lrintf(v);
}
__global__ void mg_fixk(const mg_fix *L, const uint32_t *cnt, double b, double ca, double cr, double cs, uint8_t *out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= *cnt) return;
    out[L[i].k] = mf_final(L[i].s, 1, b, ca, cr, cs);
}
enum { MG_TX = 32, MG_TY = 8 };
/* the global byte of the plane block's element e (plane k of NP, row yy, column xx of the SY x SX block) for output plane pz:
   G 0: pc ps sp w of plane z; G 1: sp of z, pc ps of l and l + 1 (l = min(z, D - 2)); G 2: pc ps of z (z clamped to the box) */
template <int G, int SY, int SX>
__device__ __forceinline__ uint8_t mg_byte(const mg_pl &P, const mg_geo &g, long long pz, long long ylo, long long xlo, int e) {
    const int k = e / (SY * SX), r = e % (SY * SX), yy = r / SX, xx = r % SX; const long long gy = ylo + yy, gx = xlo + xx;
    if (gy >= g.H || gx >= g.W) return 0;
    const long long gz = mg_cl(pz, 0, g.D - 1), l = g.D > 1 ? (gz > g.D - 2 ? g.D - 2 : gz) : 0;
    const uint8_t *src; long long z;
    if (G == 0) { src = P.p[k]; z = gz; }
    else if (G == 1) { src = k == 0 ? P.p[2] : P.p[(k - 1) & 1]; z = k == 0 ? gz : k < 3 ? l : (g.D > 1 ? l + 1 : l); }
    else { src = P.p[k]; z = gz; }
    return src[((size_t)(z - g.gzs) * g.ty + (size_t)(gy - g.gys)) * g.W + gx];
}
template <int G, int R>
__global__ void __launch_bounds__(MG_TX * MG_TY) mg_slide(mg_pl P, const int32_t *ampq, mg_geo g, int32_t *S, double b, double ca, double cr, double cs, uint8_t *out,
                                                          mg_fix *L, uint32_t *cnt, uint32_t cap, int exact) {
    constexpr int RY = MG_TY + 2 * R, RX = MG_TX + 2 * R, NW = 2 * R + 1, SY = RY + 1, SX = RX + 1, NT = MG_TX * MG_TY;
    constexpr int NP = G == 0 ? 4 : G == 1 ? 5 : 2, NE = NP * SY * SX, NL = (NE + NT - 1) / NT;
    __shared__ int32_t base[4][RY][RX], xs[4][RY][MG_TX];
    __shared__ uint8_t bq[NP][SY][SX];   /* G 0: pc ps sp w; G 1: sp, pc ps of planes l, l + 1; G 2: pc ps */
    uint8_t *const bqf = &bq[0][0][0];
    const int tx = threadIdx.x, ty = threadIdx.y, t = ty * MG_TX + tx;
    const long long x0 = (long long)blockIdx.x * MG_TX, oyb = (long long)blockIdx.y * MG_TY;
    const long long x = x0 + tx; const int oy = (int)(oyb + ty); const bool mine = x < g.W && oy < g.nyo;
    const size_t n = (size_t)g.nzo * g.nyo * g.W;
    /* the region's clamped start; the loaded block reaches one past the region (the neighbour products) */
    const long long ylo = mg_cl(g.oy0 + oyb - R, 0, g.H > 1 ? g.H - 2 : 0), xlo = mg_cl(x0 - R, 0, g.W > 1 ? g.W - 2 : 0);
    int32_t ring[4][NW], sum[4];
#pragma unroll
    for (int c = 0; c < 4; c++) { sum[c] = 0;
#pragma unroll
        for (int k = 0; k < NW; k++) ring[c][k] = 0; }
    /* the next plane's bytes are fetched into registers while this plane is summed (latency: one plane step had nothing else to do) */
    uint8_t nx[NL];
    const long long pz0 = g.oz0 - R, pz1 = g.oz0 + g.nzo + R;
#pragma unroll
    for (int q = 0; q < NL; q++) { const int e = t + q * NT; if (e < NE) bqf[e] = mg_byte<G, SY, SX>(P, g, pz0, ylo, xlo, e); }
    for (long long pz = pz0; pz < pz1; pz++) {
        if (pz + 1 < pz1) {
#pragma unroll
            for (int q = 0; q < NL; q++) { const int e = t + q * NT; if (e < NE) nx[q] = mg_byte<G, SY, SX>(P, g, pz + 1, ylo, xlo, e); }
        }
        __syncthreads();
        for (int e = t; e < RY * RX; e += MG_TX * MG_TY) {
            const int yy = e / RX, xx = e % RX;
            const long long gy = mg_cl(g.oy0 + oyb + yy - R, 0, g.H - 1), gx = mg_cl(x0 + xx - R, 0, g.W - 1);
            const int sy = (int)(gy - ylo), sx = (int)(gx - xlo);
            if (G == 0) {
                const int m = bq[3][sy][sx] > 0, sp = bq[2][sy][sx];
                base[0][yy][xx] = m ? ampq[bq[0][sy][sx] * 256 + bq[1][sy][sx]] : 0; base[1][yy][xx] = m; base[2][yy][xx] = sp; base[3][yy][xx] = sp > 0;
            } else if (G == 1) {
                const int sp = bq[0][sy][sx]; base[0][yy][xx] = sp; base[1][yy][xx] = sp > 0;
                const int ar = bq[1][sy][sx] - 128, ai = bq[2][sy][sx] - 128, br = bq[3][sy][sx] - 128, bi = bq[4][sy][sx] - 128;
                const bool ok = g.D > 1; base[2][yy][xx] = ok ? br * ar + bi * ai : 0; base[3][yy][xx] = ok ? bi * ar - br * ai : 0;
            } else {
                const int py = (int)((gy > g.H - 2 ? g.H - 2 : gy) - ylo), px = (int)((gx > g.W - 2 ? g.W - 2 : gx) - xlo);
                const int ar = bq[0][py][sx] - 128, ai = bq[1][py][sx] - 128, br = bq[0][py + 1][sx] - 128, bi = bq[1][py + 1][sx] - 128;
                const int cr_ = bq[0][sy][px] - 128, ci_ = bq[1][sy][px] - 128, dr = bq[0][sy][px + 1] - 128, di = bq[1][sy][px + 1] - 128;
                const bool oky = g.H > 1, okx = g.W > 1;
                base[0][yy][xx] = oky ? br * ar + bi * ai : 0; base[1][yy][xx] = oky ? bi * ar - br * ai : 0;
                base[2][yy][xx] = okx ? dr * cr_ + di * ci_ : 0; base[3][yy][xx] = okx ? di * cr_ - dr * ci_ : 0;
            }
        }
        __syncthreads();   /* (bq is free from here: the next plane's bytes go in) */
        if (pz + 1 < pz1) {
#pragma unroll
            for (int q = 0; q < NL; q++) { const int e = t + q * NT; if (e < NE) bqf[e] = nx[q]; }
        }
        for (int yy = ty; yy < RY; yy += MG_TY) {
#pragma unroll
            for (int c = 0; c < 4; c++) { int32_t s = 0;
#pragma unroll
                for (int d = 0; d < NW; d++) s += base[c][yy][tx + d];
                xs[c][yy][tx] = s; }
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int32_t f = 0;
#pragma unroll
            for (int d = 0; d < NW; d++) f += xs[c][ty + d][tx];
            sum[c] += f - ring[c][0];
#pragma unroll
            for (int k = 0; k < NW - 1; k++) ring[c][k] = ring[c][k + 1];
            ring[c][NW - 1] = f;
        }
        if (pz >= g.oz0 + R && mine) {   /* the window of output row pz - R is complete */
            const int oz = (int)(pz - R - g.oz0); const size_t k = ((size_t)oz * g.nyo + oy) * g.W + x;
            if (G < 2) {
#pragma unroll
                for (int c = 0; c < 4; c++) S[(size_t)(4 * G + c) * n + k] = sum[c];
            } else {
                int32_t v[MF_NS];
#pragma unroll
                for (int c = 0; c < 8; c++) v[c] = S[(size_t)c * n + k];
#pragma unroll
                for (int c = 0; c < 4; c++) v[8 + c] = sum[c];
                const size_t at = ((size_t)(g.oz0 + oz - g.gzs) * g.ty + (size_t)(g.oy0 + oy - g.gys)) * g.W + x;
                if (P.p[3][at] == 0) out[k] = 0;
                else if (exact) out[k] = mf_final(v, 1, b, ca, cr, cs);
                else { const int m = mg_final(v, b, ca, cr, cs);
                    if (m >= 0) out[k] = (uint8_t)m;
                    else { const uint32_t j = atomicAdd(cnt, 1u); out[k] = 0; if (j < cap) { L[j].k = (uint32_t)k;
#pragma unroll
                        for (int c = 0; c < MF_NS; c++) L[j].s[c] = v[c]; } } }
            }
        }
    }   /* (xs is next written after two barriers, base after one past its last read: no barrier needed here) */
}

static double g_prof_ms, g_prof_k[3], g_nfix;   /* mgpu: process (UFSM_MF_PROF timing) */
extern "C" void mf_gpu_close(mf_gpu *g);
extern "C" mf_gpu *mf_gpu_open(int dev, long long H, long long W, double mem_gb) {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || dev < 0 || dev >= n) { cudaGetLastError(); return nullptr; }
    if (cudaSetDevice(dev) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    /* bytes per tile: input rows (ZS + 8) x (YS + 8) x W x 4 planes, output ZS x YS x W x (8 sums + 1 + the fp64 list's 1 / 32) */
    const double per_y = (double)W * ((MG_ZS + 2 * MG_HALO) * 4.0 + MG_ZS * (4.0 * 8 + 1 + sizeof(mg_fix) / 32.0));
    long long ys = (long long)((mem_gb * 1e9 - 1e6) / per_y) - 2 * MG_HALO; if (ys > H) ys = H;
    if (ys < 16) { fprintf(stderr, "merge-flag gpu: %.1f GB is too little for rows of %lld cells\n", mem_gb, W); return nullptr; }
    mf_gpu *g = (mf_gpu *)calloc(1, sizeof *g); g->dev = dev; g->H = H; g->W = W; g->ys = ys;
    g->cap_in = (size_t)(MG_ZS + 2 * MG_HALO) * (ys + 2 * MG_HALO) * W; g->cap_out = (size_t)MG_ZS * ys * W;
    bool bad = false;
    for (int k = 0; k < 4; k++) bad |= cudaMalloc(&g->pl[k], g->cap_in) != cudaSuccess;
    bad |= cudaMalloc(&g->s, g->cap_out * 4 * 8) != cudaSuccess; bad |= cudaMalloc(&g->out, g->cap_out) != cudaSuccess;
    bad |= cudaMalloc(&g->ampq, 65536 * 4) != cudaSuccess;
    g->fcap = (uint32_t)(g->cap_out / 32 + 1024); bad |= cudaMalloc(&g->fix, (size_t)g->fcap * sizeof(mg_fix)) != cudaSuccess; bad |= cudaMalloc(&g->cnt, 4) != cudaSuccess; bad |= cudaMalloc(&g->cc, 16) != cudaSuccess;
    if (!bad) {
        int32_t *h = (int32_t *)malloc(65536 * 4); for (int u = 0; u < 256; u++) for (int v = 0; v < 256; v++) h[u * 256 + v] = mf_ampq(u, v);
        bad = cudaMemcpy(g->ampq, h, 65536 * 4, cudaMemcpyHostToDevice) != cudaSuccess; free(h);
    }
    if (bad) { fprintf(stderr, "merge-flag gpu: cannot allocate the tiles (%.1f GB)\n", mem_gb); cudaGetLastError(); mf_gpu_close(g); return nullptr; }
    return g;
}
extern "C" void mf_gpu_close(mf_gpu *g) {
    if (!g) return;
    if (getenv("UFSM_MF_PROF")) fprintf(stderr, "merge-flag gpu: kernels %.2f s (G0 %.2f G1 %.2f G2 %.2f; ys %lld; %.0f cells in fp64)\n", g_prof_ms / 1e3, g_prof_k[0] / 1e3, g_prof_k[1] / 1e3, g_prof_k[2] / 1e3, g->ys, g_nfix);
    for (int k = 0; k < 4; k++) cudaFree(g->pl[k]);
    cudaFree(g->s); cudaFree(g->out); cudaFree(g->ampq); cudaFree(g->fix); cudaFree(g->cnt); cudaFree(g->cc); free(g);
}
__global__ void mg_count(const uint8_t *out, const uint8_t *w, mg_geo g, int thr, unsigned long long *c) {   /* material cells, flagged ones */
    const size_t n = (size_t)g.nzo * g.nyo * g.W; unsigned long long m = 0, f = 0;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x) {
        const long long x = (long long)(k % g.W); const int oy = (int)(k / g.W % g.nyo), oz = (int)(k / ((size_t)g.W * g.nyo));
        if (w[((size_t)(g.oz0 + oz - g.gzs) * g.ty + (size_t)(g.oy0 + oy - g.gys)) * g.W + x]) { m++; f += out[k] >= thr; }
    }
    for (int o = 16; o; o >>= 1) { m += __shfl_down_sync(0xffffffffu, m, o); f += __shfl_down_sync(0xffffffffu, f, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&c[0], m); atomicAdd(&c[1], f); }
}
extern "C" int mf_gpu_rows(mf_gpu *g, const uint8_t *const pl[4], long long D, long long rz0, long long rnz, long long z0, long long nz, const double coef[4], int thr,
                           uint8_t *out, long long *nmat, long long *nflag) {
    *nmat = *nflag = 0; MG_CK(cudaMemset(g->cc, 0, 16));
    const long long H = g->H, W = g->W; MG_CK(cudaSetDevice(g->dev)); cudaGetLastError();
    for (long long oz0 = z0; oz0 < z0 + nz; oz0 += MG_ZS) for (long long oy0 = 0; oy0 < H; oy0 += g->ys) {
        mg_geo q; q.D = D; q.H = H; q.W = W; q.oz0 = oz0; q.oy0 = oy0;
        q.nzo = (int)(oz0 + MG_ZS < z0 + nz ? MG_ZS : z0 + nz - oz0); q.nyo = (int)(oy0 + g->ys < H ? g->ys : H - oy0);
        q.gzs = oz0 - MG_HALO < 0 ? 0 : oz0 - MG_HALO; const long long gze = oz0 + q.nzo + MG_HALO > D ? D : oz0 + q.nzo + MG_HALO;
        q.gys = oy0 - MG_HALO < 0 ? 0 : oy0 - MG_HALO; const long long gye = oy0 + q.nyo + MG_HALO > H ? H : oy0 + q.nyo + MG_HALO;
        q.tz = (int)(gze - q.gzs); q.ty = (int)(gye - q.gys);
        if (q.gzs < rz0 || gze > rz0 + rnz) { fprintf(stderr, "merge-flag gpu: rows %lld..%lld not in the given %lld..%lld\n", q.gzs, gze, rz0, rz0 + rnz); return 1; }
        for (int k = 0; k < 4; k++)
            MG_CK(cudaMemcpy2D(g->pl[k], (size_t)q.ty * W, pl[k] + ((size_t)(q.gzs - rz0) * H + q.gys) * W, (size_t)H * W, (size_t)q.ty * W, q.tz, cudaMemcpyHostToDevice));
        mg_pl P = {{g->pl[0], g->pl[1], g->pl[2], g->pl[3]}};
        const dim3 blk(MG_TX, MG_TY), grd((unsigned)((W + MG_TX - 1) / MG_TX), (unsigned)((q.nyo + MG_TY - 1) / MG_TY));
        cudaEvent_t e0, e1; const int prof = getenv("UFSM_MF_PROF") != nullptr; if (prof) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); }
        cudaEvent_t ek[3]; if (prof) for (int k = 0; k < 3; k++) cudaEventCreate(&ek[k]);
        mg_slide<0, 4><<<grd, blk>>>(P, g->ampq, q, g->s, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0); MG_CK(cudaGetLastError()); if (prof) cudaEventRecord(ek[0]);
        mg_slide<1, 2><<<grd, blk>>>(P, g->ampq, q, g->s, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0); MG_CK(cudaGetLastError()); if (prof) cudaEventRecord(ek[1]);
        MG_CK(cudaMemset(g->cnt, 0, 4));
        mg_slide<2, 2><<<grd, blk>>>(P, g->ampq, q, g->s, coef[0], coef[1], coef[2], coef[3], g->out, g->fix, g->cnt, g->fcap, 0);
        uint32_t nf = 0; MG_CK(cudaMemcpy(&nf, g->cnt, 4, cudaMemcpyDeviceToHost)); g_nfix += nf;
        if (nf > g->fcap)   /* (more cells near a rounding edge than the list holds: the whole tile in fp64) */
            mg_slide<2, 2><<<grd, blk>>>(P, g->ampq, q, g->s, coef[0], coef[1], coef[2], coef[3], g->out, g->fix, g->cnt, g->fcap, 1);
        else if (nf) mg_fixk<<<(nf + 255) / 256, 256>>>(g->fix, g->cnt, coef[0], coef[1], coef[2], coef[3], g->out);
        if (prof) cudaEventRecord(ek[2]);
        mg_count<<<1024, 256>>>(g->out, g->pl[3], q, thr, g->cc);
        MG_CK(cudaGetLastError());
        if (prof) { cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); g_prof_ms += ms; float m0, m1, m2; cudaEventElapsedTime(&m0, e0, ek[0]); cudaEventElapsedTime(&m1, ek[0], ek[1]); cudaEventElapsedTime(&m2, ek[1], ek[2]); g_prof_k[0] += m0; g_prof_k[1] += m1; g_prof_k[2] += m2; cudaEventDestroy(e0); cudaEventDestroy(e1); }
        MG_CK(cudaMemcpy2D(out + ((size_t)(oz0 - z0) * H + oy0) * W, (size_t)H * W, g->out, (size_t)q.nyo * W, (size_t)q.nyo * W, q.nzo, cudaMemcpyDeviceToHost));
    }
    unsigned long long c2[2]; MG_CK(cudaMemcpy(c2, g->cc, 16, cudaMemcpyDeviceToHost)); *nmat = (long long)c2[0]; *nflag = (long long)c2[1];
    return 0;
}

/* pinned host buffers (the planes' and mf's rows: page-locked copies run at the bus rate); nullptr: none (the caller mallocs) */
extern "C" void *mf_gpu_host_alloc(size_t n) { void *p = nullptr; if (cudaHostAlloc(&p, n, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); return nullptr; } return p; }
extern "C" void mf_gpu_host_free(void *p) { if (p) cudaFreeHost(p); }
