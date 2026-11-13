/* the fibre planes of a box on the GPU (src/fibregpu.h): fr_box's stamp (the min distance code per family over every step within its
   reach) and the per-voxel combine with the CT air test, fused, one thread per voxel, one CUDA block per 8^3 voxel block. The steps
   are binned to the 8^3 blocks their reach cube meets (host: a counting sort, ~steps x 8..27 entries); a block reads its list through
   shared memory. A block without steps writes the empty-voxel value (air 0, else 255) directly. Built with FMA contraction off (CMakeLists.txt UFSM_STRICT_UNITS):
   dz^2 + dy^2 + dx^2 rounds as gcc -std=c23's (no contraction), sqrt is IEEE, so the codes are the CPU's. One context (device buffers
   grown on demand) behind a mutex: the sampler's workers take turns. */
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fibregpu.h"
#include <time.h>
static double fg_t(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

#define FG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "fibre gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); rc = -1; goto out; } } while (0)
enum { FBX = 8, FBY = 4, FBZ = 4, FCH = 128 };   /* voxel block 4 x 4 x 8 (z y x): 2026-11-11, 8^3 blocks tested 3x more out-of-cube pairs */
/* exclusive sum of n int32 counts (the bin offsets; off[n - 1] = the total when cnt[n - 1] = 0): three passes over 4096-count tiles,
   tile sums, their scan in one block, each tile's scan plus its base. Integer, so identical to the cub::DeviceScan::ExclusiveSum it
   replaces (2026-11-13: CCCL 13.x's cuda/std/string_view does not compile under clang 22, the CUDA compiler since buildsys-20261113). */
enum { FSC_T = 256, FSC_I = 16, FSC_TILE = FSC_T * FSC_I };
__device__ static int fsc_block_excl(int v, int *sw, int *tot) {   /* exclusive scan of v over the 256 threads, the block total in *tot */
    const int t = threadIdx.x, l = t & 31, w = t >> 5; int x = v;
    for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, d); if (l >= d) x += y; }
    if (l == 31) sw[w] = x;
    __syncthreads();
    if (w == 0) { int s = l < FSC_T / 32 ? sw[l] : 0; for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, s, d); if (l >= d) s += y; } if (l < FSC_T / 32) sw[l] = s; }
    __syncthreads();
    const int r = x - v + (w ? sw[w - 1] : 0); *tot = sw[FSC_T / 32 - 1];
    __syncthreads();
    return r;
}
__global__ static void fsc_sums_k(const int32_t *__restrict__ cnt, long n, int32_t *part) {
    __shared__ int sw[FSC_T / 32]; const long b = (long)blockIdx.x * FSC_TILE; int s = 0, tot;
    for (int k = 0; k < FSC_I; k++) { const long i = b + k * FSC_T + threadIdx.x; if (i < n) s += cnt[i]; }
    fsc_block_excl(s, sw, &tot);
    if (threadIdx.x == 0) part[blockIdx.x] = tot;
}
__global__ static void fsc_parts_k(int32_t *part, long np) {   /* one block: part[] -> its exclusive scan, in 256-part rounds */
    __shared__ int sw[FSC_T / 32]; int carry = 0, tot;
    for (long r = 0; r < np; r += FSC_T) {
        const long i = r + threadIdx.x; const int v = i < np ? part[i] : 0; const int e = fsc_block_excl(v, sw, &tot);
        if (i < np) part[i] = carry + e;
        carry += tot;
    }
}
__global__ static void fsc_tiles_k(const int32_t *__restrict__ cnt, long n, const int32_t *__restrict__ part, int32_t *off) {
    __shared__ int sw[FSC_T / 32]; const long b = (long)blockIdx.x * FSC_TILE; int carry = part[blockIdx.x], tot;
    for (int k = 0; k < FSC_I; k++) {
        const long i = b + k * FSC_T + threadIdx.x; const int v = i < n ? cnt[i] : 0; const int e = fsc_block_excl(v, sw, &tot);
        if (i < n) off[i] = carry + e;
        carry += tot;
    }
}
static size_t fsc_tmp_bytes(long n) { return (size_t)((n + FSC_TILE - 1) / FSC_TILE) * sizeof(int32_t); }
static cudaError_t fsc_exclusive_sum(void *tmp, const int32_t *cnt, int32_t *off, long n, cudaStream_t st) {
    const long np = (n + FSC_TILE - 1) / FSC_TILE; int32_t *part = (int32_t *)tmp;
    if (np <= 0) return cudaSuccess;
    fsc_sums_k<<<(unsigned)np, FSC_T, 0, st>>>(cnt, n, part);
    fsc_parts_k<<<1, FSC_T, 0, st>>>(part, np);
    fsc_tiles_k<<<(unsigned)np, FSC_T, 0, st>>>(cnt, n, part, off);
    return cudaGetLastError();
}
typedef struct { double p[3]; float q[3]; int16_t lo[3], hi[3]; int32_t fam; } fg_step;   /* q: p - the box origin in float (the fast class test) */   /* the device's step: the reach cube box-relative, as fr_stamp's ceil / floor */   /* voxel block edge; steps per shared-memory chunk */

__global__ void k_fibre(const fg_step *st, const int32_t *boff, const int32_t *bidx, int3 nb, long long so0, long long so1, long long so2, int n0, int n1, int n2,
                        double R, double Rx, int cr, int cg, int ring, const uint8_t *ct, int iso, uint8_t *o0, uint8_t *o1) {
    __shared__ float sq[FCH][3]; __shared__ int sf[FCH], si[FCH]; __shared__ int16_t sb[FCH][6];
    const int b = blockIdx.x, bx = b % nb.x, by = b / nb.x % nb.y, bz = b / (nb.x * nb.y), t = threadIdx.x;
    const int lx = bx * FBX + (t & 7), ly = by * FBY + ((t >> 3) & 3), lz = bz * FBZ + (t >> 5), in = lx < n2 && ly < n1 && lz < n0;
    const long long gz = so0 + lz, gy = so1 + ly, gx = so2 + lx;
    const double zd = (double)gz, yd = (double)gy, xd = (double)gx, RR = __dmul_rn(R, R), RRx = __dmul_rn(Rx, Rx);
    /* per family the class of the nearest step's code: 0 <= cr (the radius), 1 <= cg (+ guard), 2 stamped (within the reach), 3 none.
       The planes depend on the codes only through these classes. Each pair is classed in float with a margin (dz / dy / dx from the
       double differences: the float dd, from q = p - the box origin in float and the voxel's box index, is within ~3e-3 of the double one for dd <= 225, boxes <= 4096) and in exact double arithmetic, fr_stamp's
       code (int)(16 sqrt(dd)), only within the margin of a class boundary or the reach. */
    int L[4] = {3, 3, 3, 3};
    const float A = (float)(((cr + 1) / 16.0) * ((cr + 1) / 16.0)), Bq = (float)(((cg + 1) / 16.0) * ((cg + 1) / 16.0)), mg = 1e-2f;   /* (A, Bq exact: dyadic) */
    const float rf = (float)RR, rfx = (float)RRx;
    const int s0 = boff[b], s1 = boff[b + 1];
    for (int c0 = s0; c0 < s1; c0 += FCH) {
        const int m = s1 - c0 < FCH ? s1 - c0 : FCH;
        __syncthreads();
        for (int k = t; k < m; k += blockDim.x) { const int j = bidx[c0 + k]; const fg_step *q = &st[j]; sq[k][0] = q->q[0]; sq[k][1] = q->q[1]; sq[k][2] = q->q[2]; sf[k] = q->fam; si[k] = j; for (int e = 0; e < 3; e++) { sb[k][e] = q->lo[e]; sb[k][3 + e] = q->hi[e]; } }
        __syncthreads();
        if (!in) continue;
        for (int k = 0; k < m; k++) {
            if (lz < sb[k][0] || lz > sb[k][3] || ly < sb[k][1] || ly > sb[k][4] || lx < sb[k][2] || lx > sb[k][5]) continue;   /* (fr_stamp's cube) */
            const int f = sf[k];
            const float zf = (float)lz - sq[k][0], yf = (float)ly - sq[k][1], xf = (float)lx - sq[k][2], ddf = zf * zf + yf * yf + xf * xf, rq = f == 3 ? rfx : rf;
            if (ddf > rq + mg) continue;
            int lev;
            if (ddf < rq - mg && ddf < A - mg) lev = 0;
            else if (ddf < rq - mg && ddf > A + mg && ddf < Bq - mg) lev = 1;
            else if (ddf < rq - mg && ddf > A + mg && ddf > Bq + mg) lev = 2;
            else {   /* near a boundary: fr_stamp's arithmetic */
                const fg_step *q = &st[si[k]]; const double dz = __dadd_rn(zd, -q->p[0]), dy = __dadd_rn(yd, -q->p[1]), dx = __dadd_rn(xd, -q->p[2]);
                const double a = __dadd_rn(__dmul_rn(dz, dz), __dmul_rn(dy, dy)), dd = __dadd_rn(a, __dmul_rn(dx, dx));
                if (dd > (f == 3 ? RRx : RR)) continue;
                const int cd = (int)__dmul_rn(16.0, __dsqrt_rn(dd)); lev = cd <= cr ? 0 : cd <= cg ? 1 : 2;
            }
            if (lev < L[f]) L[f] = lev;
        }
    }
    if (!in) return;
    const size_t i = ((size_t)lz * n1 + ly) * n2 + lx;
    const int h = L[0] == 0, v = L[1] == 0, air = ct && ct[i] < iso, o2 = L[2] >= 1;   /* (d2 > cr) */
    const int lm = L[0] < L[1] ? (L[0] < L[2] ? L[0] : L[2]) : (L[1] < L[2] ? L[1] : L[2]);
    const int any = lm <= 1, neg = ring && lm == 2 && L[3] >= 2;
    o0[i] = h && !v && o2 ? 254 : v && !h && o2 ? 0 : neg ? 0 : !any && air ? 0 : 255;
    o1[i] = v && !h && o2 ? 254 : h && !v && o2 ? 0 : neg ? 0 : !any && air ? 0 : 255;
}

/* the bins on the device: per step the blocks its cube meets, counted, scanned (cub), then filled (atomic cursors: the order within a
   bin varies, the classes are minima, so the planes do not) */
__global__ void k_fbin(const fg_step *st, long ns, int3 nb, int32_t *cnt, const int32_t *off, int32_t *idx) {
    const long s = (long)blockIdx.x * blockDim.x + threadIdx.x; if (s >= ns) return;
    const fg_step q = st[s];
    for (int z = q.lo[0] / FBZ; z <= q.hi[0] / FBZ; z++) for (int y = q.lo[1] / FBY; y <= q.hi[1] / FBY; y++) for (int x = q.lo[2] / FBX; x <= q.hi[2] / FBX; x++) {
        const long b = ((long)z * nb.y + y) * nb.x + x;
        if (!idx) atomicAdd(&cnt[b], 1); else idx[off[b] + atomicAdd(&cnt[b], 1)] = (int32_t)s; }
}
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_dev; static cudaStream_t g_st;
static void *g_buf[6]; static size_t g_cap[6];   /* steps, counts + offsets, indices, ct, planes, the scan's scratch */
static const int32_t *g_off;
static fg_step *g_hst; static size_t g_hstc;   /* (pinned) */
extern "C" void *fibre_gpu_host_alloc(size_t n) { void *p = nullptr; return cudaMallocHost(&p, n) == cudaSuccess ? p : nullptr; }
extern "C" void fibre_gpu_host_free(void *p) { if (p) cudaFreeHost(p); }
extern "C" void fibre_gpu_device(int dev) { pthread_mutex_lock(&g_mu); g_dev = dev; pthread_mutex_unlock(&g_mu); }
static int grow(int k, size_t n) { if (n <= g_cap[k]) return 0; if (g_buf[k]) cudaFree(g_buf[k]); g_buf[k] = nullptr; g_cap[k] = 0; if (cudaMalloc(&g_buf[k], n + (n >> 3)) != cudaSuccess) return -1; g_cap[k] = n + (n >> 3); return 0; }
/* the reach cube of step s in 8^3 block coordinates; 0 when it misses the box */
static int fg_cube(const fr_step *q, double R, double Rx, const int64_t so[3], const int64_t n[3], int lo[3], int hi[3], fg_step *g) {
    const double r = q->fam == 3 ? Rx : R;
    for (int k = 0; k < 3; k++) { const int64_t a = (int64_t)ceil(q->p[k] - r) - so[k], c = (int64_t)floor(q->p[k] + r) - so[k];   /* (fr_stamp's bounds) */
        if (c < a || c < 0 || a > n[k] - 1) return 0;
        const int a1 = a < 0 ? 0 : (int)a, c1 = c > n[k] - 1 ? (int)(n[k] - 1) : (int)c;
        if (g) { g->lo[k] = (int16_t)a1; g->hi[k] = (int16_t)c1; }
        const int e = k == 0 ? FBZ : k == 1 ? FBY : FBX; lo[k] = a1 / e; hi[k] = c1 / e; }
    if (g) { for (int k = 0; k < 3; k++) { g->p[k] = q->p[k]; g->q[k] = (float)(q->p[k] - (double)so[k]); } g->fam = q->fam; }
    return 1;
}

/* the device buffers for boxes up to edge^3 now (before the trainer's memory planner reads the free memory): the CT, the planes, the
   bin counts / offsets, and steps / bin entries for ~4M steps (grown on demand beyond) */
extern "C" int fibre_gpu_reserve(int edge) {
    const size_t nv = (size_t)edge * edge * edge, NB = nv / (FBX * FBY * FBZ) + 4096; int rc = 0;
    pthread_mutex_lock(&g_mu);
    if (cudaSetDevice(g_dev) != cudaSuccess || grow(3, nv) || grow(4, 2 * nv) || grow(1, (NB + 1) * 8) || grow(0, ((size_t)4 << 20) * sizeof(fg_step)) || grow(2, (size_t)96 << 20) || grow(5, (size_t)16 << 20)) rc = -1;
    pthread_mutex_unlock(&g_mu);
    return rc;
}
extern "C" int fibre_gpu_box(const fr_step *st, long nst, const int64_t so[3], const int64_t n[3], double R, double Rx, int cr, int cg, int ring,
                             const uint8_t *ct, int iso, uint8_t *o0, uint8_t *o1) {
    if (n[0] > 4096 || n[1] > 4096 || n[2] > 4096 || nst > (1L << 28)) return -1;
    const int nb[3] = {(int)((n[0] + FBZ - 1) / FBZ), (int)((n[1] + FBY - 1) / FBY), (int)((n[2] + FBX - 1) / FBX)}; const long NB = (long)nb[0] * nb[1] * nb[2];
    const size_t nv = (size_t)n[0] * n[1] * n[2]; int rc = 0; long tot = 0, ns = 0;
    double t00, tb; pthread_mutex_lock(&g_mu); t00 = fg_t();
    FG_CK(cudaSetDevice(g_dev));
    /* the steps whose cube meets the box (compacted, the cube box-relative) */
    if ((size_t)nst + 1 > g_hstc) { if (g_hst) cudaFreeHost(g_hst); g_hst = nullptr; g_hstc = 0; if (cudaMallocHost((void **)&g_hst, (nst + (nst >> 2) + 16) * sizeof *g_hst) != cudaSuccess) { rc = -1; goto out; } g_hstc = nst + (nst >> 2) + 16; }
    for (long s = 0; s < nst; s++) { int lo[3], hi[3]; if (fg_cube(&st[s], R, Rx, so, n, lo, hi, &g_hst[ns])) ns++; }
    tb = fg_t();
    if (!g_st) FG_CK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
    if (grow(0, (size_t)(ns ? ns : 1) * sizeof(fg_step)) || grow(1, (size_t)(NB + 1) * 8) || grow(3, ct ? nv : 1) || grow(4, 2 * nv)) { rc = -1; goto out; }
    {
        int32_t *cnt = (int32_t *)g_buf[1], *off = cnt + NB + 1;   /* (counts, then the cursors; the offsets) */
        if (ns) FG_CK(cudaMemcpyAsync(g_buf[0], g_hst, (size_t)ns * sizeof(fg_step), cudaMemcpyHostToDevice, g_st));
        FG_CK(cudaMemsetAsync(cnt, 0, (size_t)(NB + 1) * 4, g_st));
        const int3 nb3 = make_int3(nb[2], nb[1], nb[0]);
        if (ns) { k_fbin<<<(unsigned)((ns + 255) / 256), 256, 0, g_st>>>((const fg_step *)g_buf[0], ns, nb3, cnt, nullptr, nullptr); FG_CK(cudaGetLastError()); }
        const size_t tmpb = fsc_tmp_bytes(NB + 1);
        if (grow(5, tmpb)) { rc = -1; goto out; }
        FG_CK(fsc_exclusive_sum(g_buf[5], cnt, off, NB + 1, g_st));
        int32_t tot32 = 0; FG_CK(cudaMemcpyAsync(&tot32, off + NB, 4, cudaMemcpyDeviceToHost, g_st)); FG_CK(cudaStreamSynchronize(g_st)); tot = tot32;
        if (tot32 < 0) { rc = -1; goto out; }
        if (grow(2, (size_t)(tot ? tot : 1) * 4)) { rc = -1; goto out; }
        FG_CK(cudaMemsetAsync(cnt, 0, (size_t)(NB + 1) * 4, g_st));
        if (ns) { k_fbin<<<(unsigned)((ns + 255) / 256), 256, 0, g_st>>>((const fg_step *)g_buf[0], ns, nb3, cnt, off, (int32_t *)g_buf[2]); FG_CK(cudaGetLastError()); }
        g_off = off;
    }
    if (ct) FG_CK(cudaMemcpyAsync(g_buf[3], ct, nv, cudaMemcpyHostToDevice, g_st));
    {
        uint8_t *d0 = (uint8_t *)g_buf[4], *d1 = d0 + nv;
        cudaEvent_t e0 = nullptr, e1 = nullptr; const int tm = getenv("UFSM_FIBRE_TIMING") != nullptr; if (tm) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, g_st); }
        k_fibre<<<(unsigned)NB, FBX * FBY * FBZ, 0, g_st>>>((const fg_step *)g_buf[0], g_off, (const int32_t *)g_buf[2], make_int3(nb[2], nb[1], nb[0]),
                                                         (long long)so[0], (long long)so[1], (long long)so[2], (int)n[0], (int)n[1], (int)n[2], R, Rx, cr, cg, ring,
                                                         ct ? (const uint8_t *)g_buf[3] : nullptr, iso, d0, d1);
        FG_CK(cudaGetLastError());
        if (tm) { cudaEventRecord(e1, g_st); FG_CK(cudaStreamSynchronize(g_st)); float km = 0; cudaEventElapsedTime(&km, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
            fprintf(stderr, "  fibre gpu: %ld steps in the box, %ld bin entries, %.0f ms bins, %.0f ms H2D + kernel (kernel %.1f ms)\n", ns, tot, (tb - t00) * 1e3, (fg_t() - tb) * 1e3, km); }
        FG_CK(cudaMemcpyAsync(o0, d0, nv, cudaMemcpyDeviceToHost, g_st)); FG_CK(cudaMemcpyAsync(o1, d1, nv, cudaMemcpyDeviceToHost, g_st));   /* (o0 / o1 pinned: fibre_gpu_host_alloc) */
        FG_CK(cudaStreamSynchronize(g_st));
    }
out:
    pthread_mutex_unlock(&g_mu);
    return rc;
}
