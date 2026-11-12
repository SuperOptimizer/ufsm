/* the mesh codes stamp on the GPU (src/meshcodesgpu.h): meshcodes.c mc_stamp's quads and samples, the same double expressions in the same
   order (built with -fmad=false, no fast math: no contraction, IEEE sqrt / division), one thread per quad. The order-defined voxel rule
   becomes two passes over a z slab of 64-bit words (keys): pass 1 atomicMin of (order key << 8 | code), the order key being (tile of the
   read, row, column within the tile, a, b) = the CPU's walk order; pass 2 sets bit 63 where a sample's code is more than 4 codes from the
   winner's; then 0 (no sample) / 255 (conflict) / the winner's code. */
#include "meshcodesgpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "mesh codes gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
typedef unsigned long long u64;

__device__ __forceinline__ uint8_t mg_code(double q) { long long k = llround(q * 18.0) % 252; if (k < 0) k += 252; return (uint8_t)(1 + k); }
__device__ __forceinline__ int mg_adiff(uint8_t a, uint8_t b) { int d = ((int)a - (int)b) % 252; if (d < -126) d += 252; if (d >= 126) d -= 252; return d < 0 ? -d : d; }

struct mg_box { double sc; long long o[3], n[3], z0, z1; };
__global__ void k_mc_stamp(const mcg_tile *T, const float *xyz, const float *val, const uint8_t *valid, mg_box B, u64 *key, int pass) {
    const mcg_tile t = T[blockIdx.x]; const double sc = B.sc;
    const double lo[3] = {(double)(B.o[0] + B.z0) - 1, (double)B.o[1] - 1, (double)B.o[2] - 1}, hi[3] = {(double)(B.o[0] + B.z1), (double)(B.o[1] + B.n[1]), (double)(B.o[2] + B.n[2])};
    if (t.zlo * sc > hi[0] || t.zhi * sc < lo[0]) return;   /* (the tile's y / x bounds were culled on the host for the whole box) */
    const int nq = (t.nr - 1) * (t.nc - 1);
    for (int q = threadIdx.x; q < nq; q += blockDim.x) {
        const int qr = q / (t.nc - 1), qc = q % (t.nc - 1);
        const size_t i00 = (size_t)t.voff + (size_t)qr * t.nc + qc, idx[4] = {i00, i00 + 1, i00 + (size_t)t.nc, i00 + (size_t)t.nc + 1};
        if (!(valid[idx[0]] && valid[idx[1]] && valid[idx[2]] && valid[idx[3]])) continue;
        const double qv[4] = {val[idx[0]], val[idx[1]], val[idx[2]], val[idx[3]]};
        if (!(isfinite(qv[0]) && isfinite(qv[1]) && isfinite(qv[2]) && isfinite(qv[3]))) continue;
        double p[4][3], cl[3] = {1e30, 1e30, 1e30}, ch[3] = {-1e30, -1e30, -1e30};
        for (int k = 0; k < 4; k++) for (int a = 0; a < 3; a++) p[k][2 - a] = xyz[idx[k] * 3 + a] * sc;
        for (int k = 0; k < 4; k++) for (int d = 0; d < 3; d++) { if (p[k][d] < cl[d]) cl[d] = p[k][d]; if (p[k][d] > ch[d]) ch[d] = p[k][d]; }
        int outb = 0; for (int d = 0; d < 3; d++) if (cl[d] > hi[d] || ch[d] < lo[d]) outb = 1;
        if (outb) continue;
        double e1 = 0, e2 = 0;
        for (int d = 0; d < 3; d++) { e1 += (p[1][d] - p[0][d]) * (p[1][d] - p[0][d]); e2 += (p[2][d] - p[0][d]) * (p[2][d] - p[0][d]); }
        int nu = (int)ceil(sqrt(e1) / 0.5) + 1, nv = (int)ceil(sqrt(e2) / 0.5) + 1;
        if (nu > 256) nu = 256;
        if (nv > 256) nv = 256;
        const u64 qk = ((u64)blockIdx.x << 26) | ((u64)qr << 21) | ((u64)qc << 16);
        for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
            const double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0;
            long long g[3]; int in = 1;
            for (int d = 0; d < 3; d++) {
                const double qq = (1 - u) * (1 - v) * p[0][d] + u * (1 - v) * p[1][d] + (1 - u) * v * p[2][d] + u * v * p[3][d];
                g[d] = (long long)floor(qq + 0.5) - B.o[d];
                if (g[d] < (d ? 0 : B.z0) || g[d] >= (d ? B.n[d] : B.z1)) in = 0;
            }
            if (!in) continue;
            u64 *w = &key[((size_t)(g[0] - B.z0) * B.n[1] + g[1]) * B.n[2] + g[2]];
            const uint8_t code = mg_code((1 - u) * (1 - v) * qv[0] + u * (1 - v) * qv[1] + (1 - u) * v * qv[2] + u * v * qv[3]);
            if (!pass) atomicMin(w, ((qk | ((u64)a << 8) | (u64)b) << 8) | code);
            else if (mg_adiff((uint8_t)*(volatile u64 *)w, code) > 4) atomicOr(w, 1ull << 63);   /* (pass 2 changes only bit 63: the low byte reads stable) */
        }
    }
}
__global__ void k_mc_final(const u64 *key, size_t nvox, uint8_t *out) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvox; i += (size_t)gridDim.x * blockDim.x) {
        const u64 w = key[i]; out[i] = w == ~0ull ? 0 : (w >> 63) ? 255 : (uint8_t)w;
    }
}

static int g_dev = 0;
enum { MB_T, MB_XYZ, MB_VAL, MB_VALID, MB_KEY, MB_OUT, MB_N };
typedef struct { int dev; cudaStream_t st; struct { void *p; size_t cap; } b[MB_N]; } mgctx;
static __thread mgctx *t_mg;
static int mg_grow(mgctx *x, int i, size_t n) {
    if (x->b[i].cap >= n) return 0;
    if (x->b[i].p) cudaFree(x->b[i].p);
    const size_t c = n + n / 4; x->b[i].p = nullptr; x->b[i].cap = 0;
    if (cudaMalloc(&x->b[i].p, c) != cudaSuccess) { cudaGetLastError(); return -1; }
    x->b[i].cap = c; return 0;
}
extern "C" void mesh_codes_gpu_device(int dev) { g_dev = dev; }
extern "C" int mesh_codes_gpu_stamp(const mcg_tile *t, int nt, const float *xyz, const float *val, const uint8_t *valid, size_t nv, double sc,
                                    const int64_t o[3], const int64_t n[3], uint8_t *out) {
    if (nt <= 0 || nt >= (1 << 22) || !nv) return 2;
    mgctx *x = t_mg;
    if (!x) { x = t_mg = (mgctx *)calloc(1, sizeof *x); if (!x) return -1; x->dev = -1; }
    if (x->dev != g_dev) { MG_CK(cudaSetDevice(g_dev)); if (x->st) cudaStreamDestroy(x->st); MG_CK(cudaStreamCreateWithFlags(&x->st, cudaStreamNonBlocking)); x->dev = g_dev; }
    MG_CK(cudaSetDevice(x->dev));
    /* the slab: the keys of up to $UFSM_MESHCODES_GPU_MB (256) MB of voxels (8 bytes each), whole z planes */
    static size_t kmb = 0; if (!kmb) { const char *e = getenv("UFSM_MESHCODES_GPU_MB"); kmb = e && atoi(e) > 0 ? (size_t)atoi(e) : 256; }
    const size_t plane = (size_t)n[1] * n[2]; long long S = (long long)((kmb << 20) / 8 / (plane ? plane : 1)); if (S < 1) S = 1;
    if (S > n[0]) S = n[0];
    if (mg_grow(x, MB_T, (size_t)nt * sizeof *t) || mg_grow(x, MB_XYZ, nv * 12) || mg_grow(x, MB_VAL, nv * 4) || mg_grow(x, MB_VALID, nv) ||
        mg_grow(x, MB_KEY, (size_t)S * plane * 8) || mg_grow(x, MB_OUT, (size_t)S * plane)) { fprintf(stderr, "mesh codes gpu: out of device memory\n"); return -1; }
    mcg_tile *dt = (mcg_tile *)x->b[MB_T].p; float *dxyz = (float *)x->b[MB_XYZ].p, *dval = (float *)x->b[MB_VAL].p; uint8_t *dvalid = (uint8_t *)x->b[MB_VALID].p;
    u64 *dkey = (u64 *)x->b[MB_KEY].p; uint8_t *dout = (uint8_t *)x->b[MB_OUT].p;
    MG_CK(cudaMemcpyAsync(dt, t, (size_t)nt * sizeof *t, cudaMemcpyHostToDevice, x->st));
    MG_CK(cudaMemcpyAsync(dxyz, xyz, nv * 12, cudaMemcpyHostToDevice, x->st));
    MG_CK(cudaMemcpyAsync(dval, val, nv * 4, cudaMemcpyHostToDevice, x->st));
    MG_CK(cudaMemcpyAsync(dvalid, valid, nv, cudaMemcpyHostToDevice, x->st));
    mg_box B; B.sc = sc; for (int d = 0; d < 3; d++) { B.o[d] = o[d]; B.n[d] = n[d]; }
    for (long long z0 = 0; z0 < n[0]; z0 += S) {
        B.z0 = z0; B.z1 = z0 + S < n[0] ? z0 + S : n[0];
        const size_t nvox = (size_t)(B.z1 - B.z0) * plane;
        MG_CK(cudaMemsetAsync(dkey, 0xff, nvox * 8, x->st));
        k_mc_stamp<<<nt, 256, 0, x->st>>>(dt, dxyz, dval, dvalid, B, dkey, 0);
        k_mc_stamp<<<nt, 256, 0, x->st>>>(dt, dxyz, dval, dvalid, B, dkey, 1);
        const unsigned nb = (unsigned)((nvox + 255) / 256 < 65535 ? (nvox + 255) / 256 : 65535);
        k_mc_final<<<nb, 256, 0, x->st>>>(dkey, nvox, dout);
        MG_CK(cudaGetLastError());
        MG_CK(cudaMemcpyAsync(out + (size_t)z0 * plane, dout, nvox, cudaMemcpyDeviceToHost, x->st));
    }
    MG_CK(cudaStreamSynchronize(x->st));
    return 0;
}
