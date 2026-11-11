/* band_field_phase_pair on the GPU (src/bandgpu.h). Mirrors src/band.c (edt_feature_cap with d2 nullptr, band_field_phase_pair_ex
   without spacing / lo / hi) operation by operation; built with -fmad=false and without fast math (Makefile), so the float rounding is
   band.c's and the output bytes are identical (tests/test_band_pair.c).
   x pass: one thread per x line, the two nearest-seed sweeps. y / z passes: one thread per line (consecutive threads = consecutive x,
   so every q step is one coalesced row), the lower envelope in global scratch laid out [q][line], in chunks of lines. Pair kernel:
   one thread per voxel. */
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "bandgpu.h"
#include "band.h"

#define BG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "band gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define BINF 1e20f
enum { BT = 128, LCH = 32768 };   /* threads per block; lines per envelope chunk */

__device__ static inline int bg_seed(uint8_t c, int par) { const int lab = c > 0 && c < 255; const int k = lab ? (c - 1) / BAND_STEPS : 0; return lab && (k & 1) == par; }

__global__ void k_bg_x(const uint8_t *codes, int nlines, int nx, int par, int32_t *idx) {
    const long li = (long)blockIdx.x * blockDim.x + threadIdx.x; if (li >= nlines) return;
    const uint8_t *sd = codes + li * nx; int32_t *ir = idx + li * nx;
    int last = -1;
    for (int q = 0; q < nx; q++) { if (bg_seed(sd[q], par)) last = q; ir[q] = last; }
    int next = -1;
    for (int q = nx - 1; q >= 0; q--) {
        if (bg_seed(sd[q], par)) next = q;
        const int l = ir[q], best = l < 0 ? next : next < 0 ? l : (q - l <= next - q ? l : next);
        ir[q] = best < 0 ? -1 : (int32_t)(li * nx + best);   /* cap BINF: every distance kept */
    }
}

/* lines l0 .. l0 + nl of axis ax (1 y, 2 z): line t = (o, x) with o = (l0 + t) / nx; element q at base0 + q * st + x */
__global__ void k_bg_env(int32_t *idx, int ax, int ny, int nx, int L, long st, long sz, long sy, long l0, int nl, float *F, int32_t *S, int *V, float *Z) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nl) return;
    const long line = l0 + t, o = line / nx, x = line % nx;
    const long base0 = ax == 1 ? o * sz : o * sy;
    const long NL = nl;
#define Fq(q) F[(long)(q) * NL + t]
#define Sq(q) S[(long)(q) * NL + t]
#define Vq(q) V[(long)(q) * NL + t]
#define Zq(q) Z[(long)(q) * NL + t]
    for (int q = 0; q < L; q++) {   /* gather (band.c lean mode): the squared distance rebuilt from the feature index */
        const long row = base0 + q * st; const int32_t fi = idx[row + x]; float fv = BINF;
        if (fi >= 0) {
            long dy = 0, dx;
            if (ax == 1) dx = x - ((long)fi - row);
            else { const long r = (long)fi - (long)q * sz, fy = r / sy; dy = o - fy; dx = x - (r - fy * sy); }
            fv = (float)(dy * dy + dx * dx);
        }
        Fq(q) = fv; Sq(q) = fi;
    }
    /* envelope (band.c): the same float expressions in the same order */
    int k = 0; int v0 = -1;
    for (int q = 0; q < L; q++) {
        const float fq = Fq(q);
        if (fq >= BINF) continue;
        if (v0 < 0) { v0 = q; Vq(0) = q; Zq(0) = -BINF; Zq(1) = BINF; k = 0; continue; }
        float sct;
        for (;;) {
            const int r = Vq(k);
            sct = ((fq + (float)q * q) - (Fq(r) + (float)r * r)) / (2.f * (q - r));
            if (sct <= Zq(k) && k > 0) k--; else break;
        }
        if (sct <= Zq(k)) { Vq(k) = q; Zq(k + 1) = BINF; }
        else { k++; Vq(k) = q; Zq(k) = sct; Zq(k + 1) = BINF; }
    }
    if (v0 < 0) { for (int q = 0; q < L; q++) idx[base0 + q * st + x] = -1; return; }
    k = 0;
    for (int q = 0; q < L; q++) {
        while (Zq(k + 1) < q) k++;
        idx[base0 + q * st + x] = Sq(Vq(k));
    }
#undef Fq
#undef Sq
#undef Vq
#undef Zq
}

__global__ void k_bg_pair(const uint8_t *codes, const int32_t *ie, const int32_t *io, long N, int n1, long sz, long sy, float r2, uint8_t *phase) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
    phase[i] = PHASE_UNKNOWN;
    const int c = codes[i];
    if (c > 0 && c < 255) { phase[i] = 0; return; }
    if (ie[i] < 0 || io[i] < 0) return;
    const long a = ie[i], b = io[i];
    const long z = i / sz, y = (i / sy) % n1, x = i % sy;
    const long iaz = a / sz - z, iay = (a / sy) % n1 - y, iax = a % sy - x;
    const long ibz = b / sz - z, iby = (b / sy) % n1 - y, ibx = b % sy - x;
    const float de = (float)(iaz * iaz + iay * iay + iax * iax), dd = (float)(ibz * ibz + iby * iby + ibx * ibx);
    if (de > r2 || dd > r2) return;
    const int ka = (codes[a] - 1) / BAND_STEPS, kb = (codes[b] - 1) / BAND_STEPS;
    int dk = ((kb - ka) % 14 + 14) % 14; if (dk > 7) dk -= 14;
    if (dk != 1 && dk != -1) return;
    const float az = (float)iaz, ay = (float)iay, ax = (float)iax, bz = (float)ibz, by = (float)iby, bx = (float)ibx;
    const float da = sqrtf(de), db = sqrtf(dd);
    if (az * bz + ay * by + ax * bx > -0.5f * da * db) return;
    const float f = dk == 1 ? da / (da + db) : db / (da + db);
    int q = (int)lrintf(f * PHASE_PERIOD); if (q >= PHASE_PERIOD) q -= PHASE_PERIOD;
    phase[i] = (uint8_t)q;
}

/* a pool of slot contexts (device buffers grown on demand, a stream each): a caller takes a free one and gives it back, so the
   block threads (new pthreads every shard) reuse the same few buffer sets */
typedef struct { int init, busy; size_t cap, lcap; uint8_t *codes, *phase; int32_t *ie, *io; float *F, *Z; int32_t *S; int *V; cudaStream_t st; } bg_ctx;
enum { BG_MAXS = 16 };
static bg_ctx g_ctx[BG_MAXS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_slots = 2, g_dev = 0;

extern "C" void band_pair_gpu_slots(int slots, int device) { pthread_mutex_lock(&g_mu); g_slots = slots < 1 ? 1 : slots > BG_MAXS ? BG_MAXS : slots; g_dev = device; pthread_mutex_unlock(&g_mu); }
extern "C" void band_pair_gpu_free(void) {   /* every idle context's buffers (call once no block thread is inside) */
    pthread_mutex_lock(&g_mu);
    for (int k = 0; k < BG_MAXS; k++) { bg_ctx &G = g_ctx[k]; if (!G.init || G.busy) continue;
        cudaFree(G.codes); cudaFree(G.phase); cudaFree(G.ie); cudaFree(G.io); cudaFree(G.F); cudaFree(G.Z); cudaFree(G.S); cudaFree(G.V);
        cudaStreamDestroy(G.st); memset(&G, 0, sizeof G); }
    cudaGetLastError(); pthread_mutex_unlock(&g_mu);
}

static int bg_run(bg_ctx &G, const uint8_t *codes, const int n[3], float rmax, uint8_t *phase) {
    const long nz = n[0], ny = n[1], nx = n[2], N = nz * ny * nx, sy = nx, sz = ny * nx;
    if (N >= (1L << 31)) return -1;   /* int32 feature indices */
    BG_CK(cudaSetDevice(g_dev));   /* (per host thread state: set every call) */
    if (!G.init) { BG_CK(cudaStreamCreateWithFlags(&G.st, cudaStreamNonBlocking)); G.init = 1; }
    const long Lmax = nz > ny ? nz : ny, nlines = (nz * nx > ny * nx ? nz * nx : ny * nx), lc = nlines < LCH ? nlines : LCH;
    if ((size_t)N > G.cap) {
        cudaFree(G.codes); cudaFree(G.phase); cudaFree(G.ie); cudaFree(G.io); G.codes = G.phase = nullptr; G.ie = G.io = nullptr;
        BG_CK(cudaMalloc(&G.codes, N)); BG_CK(cudaMalloc(&G.phase, N)); BG_CK(cudaMalloc(&G.ie, N * 4)); BG_CK(cudaMalloc(&G.io, N * 4)); G.cap = N;
    }
    if ((size_t)((Lmax + 1) * lc) > G.lcap) {
        cudaFree(G.F); cudaFree(G.Z); cudaFree(G.S); cudaFree(G.V); G.F = G.Z = nullptr; G.S = nullptr; G.V = nullptr;
        const size_t e = (size_t)(Lmax + 1) * lc;
        BG_CK(cudaMalloc(&G.F, e * 4)); BG_CK(cudaMalloc(&G.Z, e * 4)); BG_CK(cudaMalloc(&G.S, e * 4)); BG_CK(cudaMalloc(&G.V, e * 4)); G.lcap = e;
    }
    BG_CK(cudaMemcpyAsync(G.codes, codes, N, cudaMemcpyHostToDevice, G.st));
    for (int par = 0; par < 2; par++) {
        int32_t *id = par ? G.io : G.ie;
        k_bg_x<<<(unsigned)((nz * ny + BT - 1) / BT), BT, 0, G.st>>>(G.codes, (int)(nz * ny), (int)nx, par, id);
        for (int ax = 1; ax < 3; ax++) {
            const long L = ax == 1 ? ny : nz, st = ax == 1 ? sy : sz, nl = (ax == 1 ? nz : ny) * nx;
            for (long l0 = 0; l0 < nl; l0 += lc) {
                const int m = (int)(nl - l0 < lc ? nl - l0 : lc);
                k_bg_env<<<(unsigned)((m + BT - 1) / BT), BT, 0, G.st>>>(id, ax, (int)ny, (int)nx, (int)L, st, sz, sy, l0, m, G.F, G.S, G.V, G.Z);
            }
        }
    }
    k_bg_pair<<<(unsigned)((N + 255) / 256), 256, 0, G.st>>>(G.codes, G.ie, G.io, N, (int)ny, sz, sy, rmax * rmax, G.phase);
    BG_CK(cudaGetLastError());
    BG_CK(cudaMemcpyAsync(phase, G.phase, N, cudaMemcpyDeviceToHost, G.st));
    BG_CK(cudaStreamSynchronize(G.st));
    return 0;
}

extern "C" int band_pair_gpu(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase) {
    int k = -1;
    pthread_mutex_lock(&g_mu);
    for (;;) { for (int q = 0; q < g_slots; q++) if (!g_ctx[q].busy) { k = q; break; } if (k >= 0) break; pthread_cond_wait(&g_cv, &g_mu); }
    g_ctx[k].busy = 1; pthread_mutex_unlock(&g_mu);
    const int rc = bg_run(g_ctx[k], codes, n, rmax, phase);
    if (rc) cudaGetLastError();
    pthread_mutex_lock(&g_mu); g_ctx[k].busy = 0; pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
    return rc;
}
