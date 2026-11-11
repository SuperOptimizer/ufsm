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
#include <stdlib.h>
#include "bandgpu.h"
#include "band.h"

#define BG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "band gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define BINF 1e20f
enum { BT = 128, LCH = 32768 };   /* threads per block; lines per envelope chunk (at least) */
/* lines per envelope chunk: at least LCH, else as many as keep the scratch (12 B x (L + 1) per line: S, V, Z) at ~4 B per block voxel; 2026-11-11:
   32768 lines left a 188-SM card at ~170 threads per SM and the y / z passes latency-bound (89% of the pair rule) */
static long bg_lines(long N, long L, long nlines) { long c = N / (3 * (L + 1)); if (c < LCH) c = LCH; return nlines < c ? nlines : c; }

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

/* the same x pass, one warp per line (2026-11-11: k_bg_x's thread per line read nx-strided bytes, 45 ms of a 420^3 tile's 165): the
   nearest seed at or left of q by a carried inclusive max-scan, at or right of q by a carried suffix min-scan, the same tie rule */
__global__ void k_bg_xw(const uint8_t *codes, long nlines, int nx, int par, int32_t *idx) {
    const long li = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); const int lane = threadIdx.x & 31; if (li >= nlines) return;
    const uint8_t *sd = codes + li * nx; int32_t *ir = idx + li * nx; const unsigned FULL = 0xffffffffu; const int BIG = 0x7fffffff;
    int carry = -1;
    for (int q0 = 0; q0 < nx; q0 += 32) {
        const int q = q0 + lane; int v = q < nx && bg_seed(sd[q], par) ? q : -1;
        for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(FULL, v, o); if (lane >= o && t > v) v = t; }
        if (carry > v) v = carry;
        if (q < nx) ir[q] = v;
        carry = __shfl_sync(FULL, v, 31);
    }
    carry = BIG;
    for (int q0 = (nx - 1) / 32 * 32; q0 >= 0; q0 -= 32) {
        const int q = q0 + lane; int v = q < nx && bg_seed(sd[q], par) ? q : BIG;
        for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_down_sync(FULL, v, o); if (lane + o < 32 && t < v) v = t; }
        if (carry < v) v = carry;
        if (q < nx) {
            const int l = ir[q], next = v == BIG ? -1 : v, best = l < 0 ? next : next < 0 ? l : (q - l <= next - q ? l : next);
            ir[q] = best < 0 ? -1 : (int32_t)(li * nx + best);
        }
        carry = __shfl_sync(FULL, v, 0);
    }
}

/* lines l0 .. l0 + nl of axis ax (1 y, 2 z): line t = (o, x) with o = (l0 + t) / nx; element q at base0 + q * st + x. The scratch holds
   per q only the feature index S (gathered before the line is overwritten), the envelope V and its boundaries Z; the squared distance
   of q's feature is rebuilt from S(q) wherever it is needed (2026-11-11: dropping the F array cut the scratch traffic by a quarter and
   lets the chunks hold a third more lines; the same integer / float expressions, so the same bytes) */
__device__ __forceinline__ float bg_f(int32_t fi, int ax, long q, long o, long x, long base0, long st, long sz, long sy) {
    if (fi < 0) return BINF;
    long dy = 0, dx;
    if (ax == 1) dx = x - ((long)fi - (base0 + q * st));
    else { const long r = (long)fi - q * sz, fy = r / sy; dy = o - fy; dx = x - (r - fy * sy); }
    return (float)(dy * dy + dx * dx);
}
__global__ void k_bg_env(int32_t *idx, int ax, int ny, int nx, int L, long st, long sz, long sy, long l0, int nl, float *F, int32_t *S, int *V, float *Z) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nl) return;
    const long line = l0 + t, o = line / nx, x = line % nx;
    const long base0 = ax == 1 ? o * sz : o * sy;
    const long NL = nl;
#define Sq(q) S[(long)(q) * NL + t]
#define Vq(q) V[(long)(q) * NL + t]
#define Zq(q) Z[(long)(q) * NL + t]
#define Fq(q) bg_f(Sq(q), ax, (q), o, x, base0, st, sz, sy)
    for (int q = 0; q < L; q++) Sq(q) = idx[base0 + q * st + x];
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

/* ---- volcomp lossless (mode 1) codes chunks decoded on the device (2026-11-11), byte-identical to vll_decode_sub: one CUDA block of 32
   threads per chunk (its 8 tANS tables in shared memory), one thread per substream decoding its 16 blocks' residuals (or values) into
   a per-block scratch; then one 256-thread block per 16^3 block: unzigzag, the x / y / z prefix rebuild in shared memory, the scatter
   into the box. Any failed check marks the chunk: the caller decodes on the CPU, which gives the reference's status. ---- */
struct gbr { const uint32_t *w; uint32_t lo, hi, nxt, off, left; };   /* LSB-first bit reader over 4-byte aligned words (as nn_gpudec) */
__device__ __forceinline__ void gbr_init(gbr &r, const uint8_t *p, uint32_t n) { r.w = (const uint32_t *)p; r.lo = __ldg(r.w); r.hi = __ldg(r.w + 1); r.nxt = __ldg(r.w + 2); r.w += 3; r.off = 0; r.left = n * 8u; }
__device__ __forceinline__ bool gbr_get(gbr &r, uint32_t n, uint32_t &v) {
    if (n > r.left) return false;
    v = n ? __funnelshift_r(r.lo, r.hi, r.off) & (0xffffffffu >> (32u - n)) : 0u;
    r.off += n; r.left -= n;
    if (r.off >= 32u) { r.off -= 32u; r.lo = r.hi; r.hi = r.nxt; r.nxt = __ldg(r.w++); }
    return true;
}
__device__ __forceinline__ bool gbr_finished(const gbr &r) { return r.left < 8u && (r.left == 0 || (__funnelshift_r(r.lo, r.hi, r.off) & ((1u << r.left) - 1u)) == 0); }
__device__ __forceinline__ bool ll_hyb(gbr &r, uint32_t tok, uint32_t &u) {   /* vll_hyb_read: 16 literal tokens */
    if (tok < 16u) { u = tok; return true; }
    const uint32_t k = tok - 12u; uint32_t e; if (!gbr_get(r, k, e)) return false; u = (1u << k) | e; return true;
}
__device__ __forceinline__ int ll_tok(const uint32_t *dt, uint32_t m, uint32_t *x, uint32_t &k, gbr &ra) {   /* vf_rdec_get */
    const uint32_t e = dt[m * 1024u + x[k]]; uint32_t bits;
    if (!gbr_get(ra, e >> 24, bits)) return -1;
    x[k] = (e & 0xffffu) + bits; k ^= 1u; return (int)(e >> 16 & 255u);
}
__global__ void __launch_bounds__(32) k_ll_sub(const bg_lchunk *__restrict__ ch, const uint8_t *__restrict__ blob, uint8_t *__restrict__ scr, uint8_t *__restrict__ tag, uint32_t *__restrict__ status) {
    __shared__ uint32_t dt[8 * 1024];
    const bg_lchunk &C = ch[blockIdx.x]; const uint32_t s = threadIdx.x;
    if (C.mode != 1u) { status[blockIdx.x * 32 + s] = 1; return; }
    { const uint4 *src = (const uint4 *)(blob + C.tab_off); uint4 *d4 = (uint4 *)dt; for (int q = threadIdx.x; q < 8 * 1024 / 4; q += 32) d4[q] = __ldg(src + q); }
    __syncthreads();
    gbr ra, rb; gbr_init(ra, blob + C.tok_off[s], C.tok_n[s]); gbr_init(rb, blob + C.byp_off[s], C.byp_n[s]);
    uint32_t x[2], k = 0, ok = 0;
    if (!gbr_get(ra, 10, x[0]) || !gbr_get(ra, 10, x[1])) goto done;
    for (uint32_t bi = s * 16u; bi < s * 16u + 16u; bi++) {
        uint8_t *u = scr + ((size_t)blockIdx.x * 512u + bi) * 4096u; uint8_t *tg = tag + (size_t)blockIdx.x * 512u + bi;
        const int t = ll_tok(dt, 0u, x, k, ra);
        if (t < 0 || t > 3) goto done;
        if (t == 0) { uint32_t v; if (!gbr_get(rb, 8, v)) goto done; u[0] = (uint8_t)v; *tg = 0; }   /* CONST */
        else if (t == 3) { for (int i = 0; i < 4096; i++) { uint32_t v; if (!gbr_get(rb, 8, v)) goto done; u[i] = (uint8_t)v; } *tg = 2; }   /* RAW */
        else if (t == 1) {   /* SPARSE */
            uint4 *z4 = (uint4 *)u; for (int q = 0; q < 256; q++) z4[q] = make_uint4(0, 0, 0, 0);
            uint32_t pos = 0, prev_run = 1;
            for (;;) {
                const int rt = ll_tok(dt, 1u + (prev_run == 0u ? 0u : 1u), x, k, ra);
                if (rt < 0) goto done;
                if (rt == 31) break;
                if (rt > 23) goto done;
                uint32_t run; if (!ll_hyb(rb, (uint32_t)rt, run)) goto done;
                if (pos >= 4096u || run > 4095u - pos) goto done;
                pos += run; prev_run = run;
                const int lt = ll_tok(dt, 3u + (run == 0u ? 0u : 1u), x, k, ra);
                if (lt < 0 || lt > 19) goto done;
                uint32_t uu; if (!ll_hyb(rb, (uint32_t)lt, uu)) goto done;
                if (uu > 254u) goto done;
                u[pos++] = (uint8_t)(uu + 1u);
            }
            *tg = 1;
        } else {   /* DENSE */
            for (uint32_t i = 0; i < 4096u; i++) {
                const uint32_t a = i < 256u ? 0u : u[i - 256u];
                const int lt = ll_tok(dt, 5u + (a == 0u ? 0u : a <= 2u ? 1u : 2u), x, k, ra);
                if (lt < 0 || lt > 19) goto done;
                uint32_t uu; if (!ll_hyb(rb, (uint32_t)lt, uu)) goto done;
                u[i] = (uint8_t)uu;
            }
            *tg = 1;
        }
    }
    ok = x[0] == 0 && x[1] == 0 && gbr_finished(ra) && gbr_finished(rb);
done:
    status[blockIdx.x * 32 + s] = ok;
}
__global__ void __launch_bounds__(256) k_ll_build(const bg_lchunk *__restrict__ ch, const uint8_t *__restrict__ blob, const uint8_t *__restrict__ scr, const uint8_t *__restrict__ tag,
                                                   int nz, int ny, int nx, int3 lo, int3 hi, uint8_t *__restrict__ codes) {
    __shared__ uint8_t b[4096];
    const uint32_t cb = blockIdx.x >> 9, bi = blockIdx.x & 511u, t = threadIdx.x; const bg_lchunk &C = ch[cb];
    const uint32_t bz = bi >> 6, by = (bi >> 3) & 7u, bx = bi & 7u;
    const int gz0 = C.o[0] + (int)bz * 16, gy0 = C.o[1] + (int)by * 16, gx0 = C.o[2] + (int)bx * 16;
    if (gz0 >= hi.x || gz0 + 16 <= lo.x || gy0 >= hi.y || gy0 + 16 <= lo.y || gx0 >= hi.z || gx0 + 16 <= lo.z) return;
    const uint32_t y = t >> 4, xx = t & 15u;
    if (C.mode == 3u) { for (int z = 0; z < 16; z++) b[z * 256 + t] = (uint8_t)C.value; }
    else if (C.mode == 2u) { for (int z = 0; z < 16; z++) b[z * 256 + t] = blob[C.raw_off + ((size_t)(bz * 16u + z) * 128u + (by * 16u + y)) * 128u + bx * 16u + xx]; }
    else {
        const uint8_t *u = scr + ((size_t)cb * 512u + bi) * 4096u; const uint8_t tg = tag[(size_t)cb * 512u + bi];
        if (tg == 0) { const uint8_t v = u[0]; for (int z = 0; z < 16; z++) b[z * 256 + t] = v; }
        else if (tg == 2) { for (int z = 0; z < 16; z++) b[z * 256 + t] = u[z * 256 + t]; }
        else {
            for (int z = 0; z < 16; z++) { const uint32_t v = u[z * 256 + t]; b[z * 256 + t] = (uint8_t)((v >> 1) ^ (uint8_t)(0u - (v & 1u))); }   /* VLL_UNZZ */
            __syncthreads();
            if (t == 0) { b[0] = (uint8_t)(128u + b[0]); for (int q = 1; q < 16; q++) b[q] = (uint8_t)(b[q - 1] + b[q]); }
            __syncthreads();
            if (t < 16) for (int q = 1; q < 16; q++) b[q * 16 + t] = (uint8_t)(b[(q - 1) * 16 + t] + b[q * 16 + t]);
            __syncthreads();
            for (int z = 1; z < 16; z++) b[z * 256 + t] = (uint8_t)(b[(z - 1) * 256 + t] + b[z * 256 + t]);
        }
    }
    __syncthreads();
    const int gy = gy0 + (int)y, gx = gx0 + (int)xx;
    if (gy < lo.y || gy >= hi.y || gx < lo.z || gx >= hi.z) return;
    for (int z = 0; z < 16; z++) { const int gz = gz0 + z; if (gz < lo.x || gz >= hi.x) continue; codes[((size_t)gz * ny + gy) * nx + gx] = b[z * 256 + t]; }
}

/* a pool of slot contexts (device buffers grown on demand, a stream each): a caller takes a free one and gives it back, so the
   block threads (new pthreads every shard) reuse the same few buffer sets */
typedef struct { int init, busy; size_t cap, lcap; uint8_t *codes, *phase; int32_t *ie, *io; float *F, *Z; int32_t *S; int *V; cudaStream_t st;
                 size_t bcap, ccap, scap; uint8_t *blob, *scr, *tag; bg_lchunk *ch; uint32_t *status, *hstatus; } bg_ctx;   /* (the codes decode: blob, chunk records, scratch) */
enum { BG_MAXS = 16 };
static bg_ctx g_ctx[BG_MAXS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_slots = 2, g_dev = 0;
static const int g_xw = !getenv("UFSM_BAND_X1");
static const int g_timing = getenv("UFSM_BAND_TIMING") != nullptr;   /* per-call stage times on stderr */   /* (UFSM_BAND_X1: the thread-per-line x pass, for A/B) */

extern "C" void band_pair_gpu_slots(int slots, int device) { pthread_mutex_lock(&g_mu); g_slots = slots < 1 ? 1 : slots > BG_MAXS ? BG_MAXS : slots; g_dev = device; pthread_mutex_unlock(&g_mu); }
extern "C" void band_pair_gpu_free(void) {   /* every idle context's buffers (call once no block thread is inside) */
    pthread_mutex_lock(&g_mu);
    for (int k = 0; k < BG_MAXS; k++) { bg_ctx &G = g_ctx[k]; if (!G.init || G.busy) continue;
        cudaFree(G.codes); cudaFree(G.phase); cudaFree(G.ie); cudaFree(G.io); cudaFree(G.F); cudaFree(G.Z); cudaFree(G.S); cudaFree(G.V);
        cudaFree(G.blob); cudaFree(G.scr); cudaFree(G.tag); cudaFree(G.ch); cudaFree(G.status); cudaFreeHost(G.hstatus);
        cudaStreamDestroy(G.st); memset(&G, 0, sizeof G); }
    cudaGetLastError(); pthread_mutex_unlock(&g_mu);
}

static int bg_alloc(bg_ctx &G, long N, long Lmax, long lc) {
    if (!G.init) { BG_CK(cudaStreamCreateWithFlags(&G.st, cudaStreamNonBlocking)); G.init = 1; }
    if ((size_t)N > G.cap) {
        cudaFree(G.codes); cudaFree(G.phase); cudaFree(G.ie); cudaFree(G.io); G.codes = G.phase = nullptr; G.ie = G.io = nullptr;
        BG_CK(cudaMalloc(&G.codes, N)); BG_CK(cudaMalloc(&G.phase, N)); BG_CK(cudaMalloc(&G.ie, N * 4)); BG_CK(cudaMalloc(&G.io, N * 4)); G.cap = N;
    }
    if ((size_t)((Lmax + 1) * lc) > G.lcap) {
        cudaFree(G.F); cudaFree(G.Z); cudaFree(G.S); cudaFree(G.V); G.F = G.Z = nullptr; G.S = nullptr; G.V = nullptr;
        const size_t e = (size_t)(Lmax + 1) * lc;
        BG_CK(cudaMalloc(&G.Z, e * 4)); BG_CK(cudaMalloc(&G.S, e * 4)); BG_CK(cudaMalloc(&G.V, e * 4)); G.lcap = e;
    }
    return 0;
}
static int bg_fields(bg_ctx &G, const int n[3], float rmax, cudaEvent_t *ev);
static int bg_run(bg_ctx &G, const uint8_t *codes, const int n[3], float rmax, uint8_t *phase) {
    const long nz = n[0], ny = n[1], nx = n[2], N = nz * ny * nx;
    if (N >= (1L << 31)) return -1;   /* int32 feature indices */
    BG_CK(cudaSetDevice(g_dev));   /* (per host thread state: set every call) */
    const long Lmax = nz > ny ? nz : ny, nlines = (nz * nx > ny * nx ? nz * nx : ny * nx), lc = bg_lines(N, Lmax, nlines);
    if (bg_alloc(G, N, Lmax, lc)) return -1;
    cudaEvent_t ev[6]; const int tim = g_timing; if (tim) for (int k = 0; k < 6; k++) cudaEventCreate(&ev[k]);
    if (tim) cudaEventRecord(ev[0], G.st);
    BG_CK(cudaMemcpyAsync(G.codes, codes, N, cudaMemcpyHostToDevice, G.st));
    if (tim) cudaEventRecord(ev[1], G.st);
    if (bg_fields(G, n, rmax, tim ? ev : nullptr)) return -1;
    BG_CK(cudaMemcpyAsync(phase, G.phase, N, cudaMemcpyDeviceToHost, G.st));
    if (tim) cudaEventRecord(ev[5], G.st);
    BG_CK(cudaStreamSynchronize(G.st));
    if (tim) { float t[5]; for (int k = 0; k < 5; k++) cudaEventElapsedTime(&t[k], ev[k], ev[k + 1]);
        fprintf(stderr, "band gpu: %ldx%ldx%ld: H2D %.1f ms, parity 0 %.1f, parity 1 %.1f, pair %.1f, D2H %.1f\n", nz, ny, nx, t[0], t[1], t[2], t[3], t[4]);
        for (int k = 0; k < 6; k++) cudaEventDestroy(ev[k]); }
    return 0;
}
/* the two parity feature transforms and the pair kernel on G.codes (n voxels) into G.phase; ev (optional): records 2 (parity 1
   starts), 3 (the pair kernel starts), 4 (done) */
static int bg_fields(bg_ctx &G, const int n[3], float rmax, cudaEvent_t *ev) {
    const long nz = n[0], ny = n[1], nx = n[2], N = nz * ny * nx, sy = nx, sz = ny * nx;
    const long Lmax = nz > ny ? nz : ny, nlines = (nz * nx > ny * nx ? nz * nx : ny * nx), lc = bg_lines(N, Lmax, nlines);
    const int tim = ev != nullptr;
    for (int par = 0; par < 2; par++) {
        int32_t *id = par ? G.io : G.ie;
        if (tim && par) cudaEventRecord(ev[2], G.st);
        if (g_xw) k_bg_xw<<<(unsigned)((nz * ny + 7) / 8), 256, 0, G.st>>>(G.codes, nz * ny, (int)nx, par, id);
        else k_bg_x<<<(unsigned)((nz * ny + BT - 1) / BT), BT, 0, G.st>>>(G.codes, (int)(nz * ny), (int)nx, par, id);
        for (int ax = 1; ax < 3; ax++) {
            const long L = ax == 1 ? ny : nz, st = ax == 1 ? sy : sz, nl = (ax == 1 ? nz : ny) * nx;
            for (long l0 = 0; l0 < nl; l0 += lc) {
                const int m = (int)(nl - l0 < lc ? nl - l0 : lc);
                k_bg_env<<<(unsigned)((m + BT - 1) / BT), BT, 0, G.st>>>(id, ax, (int)ny, (int)nx, (int)L, st, sz, sy, l0, m, G.F, G.S, G.V, G.Z);
            }
        }
    }
    if (tim) cudaEventRecord(ev[3], G.st);
    k_bg_pair<<<(unsigned)((N + 255) / 256), 256, 0, G.st>>>(G.codes, G.ie, G.io, N, (int)ny, sz, sy, rmax * rmax, G.phase);
    BG_CK(cudaGetLastError());
    if (tim) cudaEventRecord(ev[4], G.st);
    return 0;
}

extern "C" void *band_pair_gpu_host_alloc(size_t n) { void *p = nullptr; if (cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return nullptr; } return p; }
extern "C" void band_pair_gpu_host_free(void *p) { if (p) cudaFreeHost(p); }
extern "C" int band_pair_gpu_reserve(int edge) {   /* all g_slots contexts sized for edge^3 blocks (so a memory planner sees them) */
    const long E = edge, N = E * E * E, lines = E * E, lc = bg_lines(N, E, lines); int rc = 0;
    pthread_mutex_lock(&g_mu);
    if (cudaSetDevice(g_dev) != cudaSuccess) rc = -1;
    for (int k = 0; k < g_slots && !rc; k++) if (!g_ctx[k].busy) rc = bg_alloc(g_ctx[k], N, E, lc);
    pthread_mutex_unlock(&g_mu);
    if (rc) cudaGetLastError();
    return rc;
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

/* codes decoded on the device in batches of chunks (the scratch: 4 KB per 16^3 block, at most ~1 B per box voxel), the fields, the crop
   down into out */
static int bg_codes_run(bg_ctx &G, const bg_lchunk *ch, int nch, const uint8_t *blob, size_t blob_n, const int n[3], const int lo[3], const int hi[3], float rmax,
                        const int co[3], const int cn[3], uint8_t *out) {   /* rmax <= 0: the decoded codes of the crop instead (tests) */
    const long nz = n[0], ny = n[1], nx = n[2], N = nz * ny * nx;
    if (N >= (1L << 31)) return -1;
    BG_CK(cudaSetDevice(g_dev));
    const long Lmax = nz > ny ? nz : ny, nlines = (nz * nx > ny * nx ? nz * nx : ny * nx), lc = bg_lines(N, Lmax, nlines);
    if (bg_alloc(G, N, Lmax, lc)) return -1;
    long cbat = N / (2L << 20); if (cbat < 8) cbat = 8; if (cbat > nch) cbat = nch > 0 ? nch : 1;
    if (blob_n + 64 > G.bcap) { cudaFree(G.blob); G.blob = nullptr; BG_CK(cudaMalloc(&G.blob, blob_n + 64)); G.bcap = blob_n + 64; }
    if ((size_t)nch > G.ccap) { cudaFree(G.ch); cudaFree(G.status); cudaFreeHost(G.hstatus); G.ch = nullptr; G.status = nullptr; G.hstatus = nullptr;
        BG_CK(cudaMalloc(&G.ch, (size_t)nch * sizeof(bg_lchunk))); BG_CK(cudaMalloc(&G.status, (size_t)nch * 32 * 4)); BG_CK(cudaHostAlloc(&G.hstatus, (size_t)nch * 32 * 4, cudaHostAllocPortable)); G.ccap = nch; }
    if ((size_t)cbat > G.scap) { cudaFree(G.scr); cudaFree(G.tag); G.scr = G.tag = nullptr; BG_CK(cudaMalloc(&G.scr, (size_t)cbat * 512 * 4096)); BG_CK(cudaMalloc(&G.tag, (size_t)cbat * 512)); G.scap = cbat; }
    cudaEvent_t ev[6]; const int tim = g_timing; if (tim) for (int k = 0; k < 6; k++) cudaEventCreate(&ev[k]);
    if (tim) cudaEventRecord(ev[0], G.st);
    BG_CK(cudaMemsetAsync(G.codes, 0, N, G.st));
    if (nch) {
        BG_CK(cudaMemcpyAsync(G.blob, blob, blob_n, cudaMemcpyHostToDevice, G.st)); BG_CK(cudaMemsetAsync(G.blob + blob_n, 0, 64, G.st));
        BG_CK(cudaMemcpyAsync(G.ch, ch, (size_t)nch * sizeof(bg_lchunk), cudaMemcpyHostToDevice, G.st));
        const int3 l3 = make_int3(lo[0], lo[1], lo[2]), h3 = make_int3(hi[0], hi[1], hi[2]);
        for (long c0 = 0; c0 < nch; c0 += cbat) {
            const int m = (int)(nch - c0 < cbat ? nch - c0 : cbat);
            k_ll_sub<<<m, 32, 0, G.st>>>(G.ch + c0, G.blob, G.scr, G.tag, G.status + c0 * 32);
            k_ll_build<<<(unsigned)m * 512u, 256, 0, G.st>>>(G.ch + c0, G.blob, G.scr, G.tag, (int)nz, (int)ny, (int)nx, l3, h3, G.codes);
        }
        BG_CK(cudaGetLastError());
        BG_CK(cudaMemcpyAsync(G.hstatus, G.status, (size_t)nch * 32 * 4, cudaMemcpyDeviceToHost, G.st));
    }
    if (tim) cudaEventRecord(ev[1], G.st);
    if (rmax > 0 && bg_fields(G, n, rmax, tim ? ev : nullptr)) return -1;
    if (rmax <= 0 && tim) for (int k = 2; k < 5; k++) cudaEventRecord(ev[k], G.st);
    cudaMemcpy3DParms P = {};
    P.srcPtr = make_cudaPitchedPtr(rmax > 0 ? G.phase : G.codes, (size_t)nx, (size_t)nx, (size_t)ny); P.srcPos = make_cudaPos((size_t)co[2], (size_t)co[1], (size_t)co[0]);
    P.dstPtr = make_cudaPitchedPtr(out, (size_t)cn[2], (size_t)cn[2], (size_t)cn[1]); P.extent = make_cudaExtent((size_t)cn[2], (size_t)cn[1], (size_t)cn[0]); P.kind = cudaMemcpyDeviceToHost;
    BG_CK(cudaMemcpy3DAsync(&P, G.st));
    if (tim) cudaEventRecord(ev[5], G.st);
    BG_CK(cudaStreamSynchronize(G.st));
    if (tim) { float t[5]; for (int k = 0; k < 5; k++) cudaEventElapsedTime(&t[k], ev[k], ev[k + 1]);
        fprintf(stderr, "band gpu codes: %ldx%ldx%ld, %d chunks: decode %.1f ms, parity 0 %.1f, parity 1 %.1f, pair %.1f, crop D2H %.1f\n", nz, ny, nx, nch, t[0], t[1], t[2], t[3], t[4]);
        for (int k = 0; k < 6; k++) cudaEventDestroy(ev[k]); }
    for (long q = 0; q < (long)nch * 32; q++) if (!G.hstatus[q]) return 1;
    return 0;
}
extern "C" int band_pair_gpu_codes(const bg_lchunk *ch, int nch, const uint8_t *blob, size_t blob_n, const int n[3], const int lo[3], const int hi[3],
                                   float rmax, const int crop_o[3], const int crop_n[3], uint8_t *out) {
    int k = -1;
    pthread_mutex_lock(&g_mu);
    for (;;) { for (int q = 0; q < g_slots; q++) if (!g_ctx[q].busy) { k = q; break; } if (k >= 0) break; pthread_cond_wait(&g_cv, &g_mu); }
    g_ctx[k].busy = 1; pthread_mutex_unlock(&g_mu);
    const int rc = bg_codes_run(g_ctx[k], ch, nch, blob, blob_n, n, lo, hi, rmax, crop_o, crop_n, out);
    if (rc < 0) cudaGetLastError();
    pthread_mutex_lock(&g_mu); g_ctx[k].busy = 0; pthread_cond_signal(&g_cv); pthread_mutex_unlock(&g_mu);
    return rc;
}
