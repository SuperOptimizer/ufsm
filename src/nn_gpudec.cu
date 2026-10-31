/* volcomp lossy substream decoding on the GPU (src/gpudec.h), byte-identical to vf_decode_sub / volcomp_fast.h.
 *
 * Built without fast math and with -fmad=false (Makefile): the dequantisation and the inverse DCT round every float operation as
 * the CPU does (gcc -std=c23 contracts nothing), in the reference's order. Two kernels per batch:
 *   k_entropy: one thread per substream. The two tANS lanes (DC and level tokens lane 0, run tokens and the EOB lane 1, as
 *     volcomp_fast.h) and the bypass bit stream, with every check of the reference; it writes per block the DC and the AC
 *     coefficients as (natural index, sign, magnitude) records. Any check that fails marks the substream: the host then decodes
 *     the whole request on the CPU, which gives the reference's status (and bytes) exactly.
 *   k_idct: one CUDA block per 16^3 block. Dequantise into a zeroed shared block, x / y / z passes of VF_DCT16_INV_BODY on every
 *     line (the reference skips lines it knows are zero: the transform of a zero line is +0, the same), round as vf_scatter_block,
 *     DC-only blocks as vf_flat_value.
 * A service thread batches the requests of all calling threads: callers reserve space in the open batch's pinned staging, copy
 * their tables and substream bytes in parallel, and wait; the service runs one batch at a time (H2D, kernels, D2H) while the next
 * one fills; callers copy their decoded substreams out of the pinned output in parallel. */
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpudec.h"

enum { NSUB = 32, BLKV = 4096, TABW = 10 * 1024, NRADII = 46, TAB_BYTES = (TABW * 4 + NRADII * 4 + 15) & ~15, SUB_BYTES = 65536 };

struct gd_item { uint32_t tok_off, tok_n, byp_off, byp_n, tab_off, rec_off, zs, pad; uint64_t dst; };   /* byte offsets in the input; rec_off in records; output: the
                                                                                            substream's (z 0, y 0, x 0) at dst, z planes zs bytes apart, rows 128 */

/* per scan position: natural index | radius << 12 | zero-axis count << 18 | band << 20 (volcomp_fast.h vfx_pos_tab) */
__constant__ float c_scale[4] = {0.0441941738f, 0.03125f, 0.0220970869f, 0.015625f};

/* ---- bit reader: LSB first over 32-bit aligned words; a two-word window (lo, hi) read by one funnel shift, the next word loaded
   ahead (its latency hides behind 32 bits of decoding). 32-bit operations only: 64-bit shifts are several dependent instructions on
   the GPU, and this is the serial chain. The reference's success / exhaustion rules (vf_br_get, vf_br_finished). n <= 31 (the
   format's widest read is 19 bits). Streams have 16+ bytes of slack after them. ---- */
struct gbr { const uint32_t *w; uint32_t lo, hi, nxt, off, left; };
__device__ __forceinline__ void gbr_init(gbr &r, const uint8_t *p, uint32_t n) {
    r.w = (const uint32_t *)p; r.lo = __ldg(r.w); r.hi = __ldg(r.w + 1); r.nxt = __ldg(r.w + 2); r.w += 3; r.off = 0; r.left = n * 8u;
}
__device__ __forceinline__ bool gbr_get(gbr &r, uint32_t n, uint32_t &v) {
    if (n > r.left) return false;
    v = __funnelshift_r(r.lo, r.hi, r.off) & ((1u << n) - 1u);
    r.off += n; r.left -= n;
    if (r.off >= 32u) { r.off -= 32u; r.lo = r.hi; r.hi = r.nxt; r.nxt = __ldg(r.w++); }
    return true;
}
/* all bytes consumed (fewer than 8 bits left) and the bits left are zero */
__device__ __forceinline__ bool gbr_finished(const gbr &r) {
    return r.left < 8u && (__funnelshift_r(r.lo, r.hi, r.off) & ((1u << r.left) - 1u)) == 0;
}
/* HybridUint (vf_hyb_read), branch free: tokens 0..3 read 0 bits (a 0-bit read always succeeds, as the reference reads none) */
__device__ __forceinline__ bool hyb(gbr &r, uint32_t tok, uint32_t &u) {
    const uint32_t k = tok < 4u ? 0u : tok - 2u; uint32_t e;
    if (!gbr_get(r, k, e)) return false;
    u = tok < 4u ? tok : (1u << k) | e; return true;
}

/* meta per item: dc[16] (int32), start[17] (record index of each block's first AC record, start[16] = end) */
enum { META = 33 };
/* one CUDA block (32 threads) per request: its chunk's 10 tANS tables in shared memory, one thread per substream */
struct gd_group { uint32_t item0, n, tab_off, pad; };
__global__ void __launch_bounds__(32) k_entropy(const gd_group *__restrict__ groups, const gd_item *__restrict__ items, const uint8_t *__restrict__ in,
                                                const uint32_t *__restrict__ pos_tab, uint32_t *__restrict__ rec, uint32_t *__restrict__ meta, uint32_t *__restrict__ status) {
    __shared__ uint32_t dt[TABW];
    const gd_group g = groups[blockIdx.x];
    { const uint4 *src = (const uint4 *)(in + g.tab_off); uint4 *d4 = (uint4 *)dt; for (int k = threadIdx.x; k < TABW / 4; k += 32) d4[k] = __ldg(src + k); }
    __syncthreads();
    if (threadIdx.x >= g.n) return;
    const int i = (int)(g.item0 + threadIdx.x);
    const gd_item it = items[i];
    uint32_t *R = rec + it.rec_off, *M = meta + (size_t)i * META, nr = 0;
    gbr ra, rb; gbr_init(ra, in + it.tok_off, it.tok_n); gbr_init(rb, in + it.byp_off, it.byp_n);
    uint32_t x0, x1, b, ok = 0;
    if (!gbr_get(ra, 10, x0) || !gbr_get(ra, 10, x1)) goto done;
    {
        int64_t prev_dc = 0;
        for (uint32_t bl = 0; bl < 16; bl++) {
            uint32_t e = dt[9 * 1024 + x0], t = e >> 16 & 255u;
            if (!gbr_get(ra, e >> 24, b)) goto done;
            x0 = (e & 0xffffu) + b;
            if (t > 20u) goto done;
            uint32_t u;
            if (!hyb(rb, t, u)) goto done;
            const int64_t dc = prev_dc + (int64_t)((int32_t)(u >> 1) ^ -(int32_t)(u & 1u));
            if (dc < -65536 || dc > 65536) goto done;
            prev_dc = dc;
            M[bl] = (uint32_t)(int32_t)dc; M[16 + bl] = nr;
            uint32_t pos = 1;
            for (;;) {
                const uint32_t band = pos < 128u ? 0u : pos < 1024u ? 1u : 2u;
                e = dt[band * 1024 + x1]; t = e >> 16 & 255u;
                if (!gbr_get(ra, e >> 24, b)) goto done;
                x1 = (e & 0xffffu) + b;
                if (t == 31u) break;
                if (t > 13u) goto done;
                uint32_t run;
                if (!hyb(rb, t, run)) goto done;
                pos += run;
                if (pos >= (uint32_t)BLKV) goto done;
                const uint32_t lband = pos < 128u ? 0u : pos < 1024u ? 1u : 2u;
                e = dt[(3u + lband * 2u + (run == 0u ? 1u : 0u)) * 1024 + x0];
                const uint32_t lt = e >> 16 & 255u;
                if (!gbr_get(ra, e >> 24, b)) goto done;
                x0 = (e & 0xffffu) + b;
                if (lt > 14u) goto done;
                /* level: tokens 0..3 are the magnitude, then a sign bit; else k extra bits and the sign in one read (branch free) */
                const uint32_t k = lt < 4u ? 0u : lt - 2u; uint32_t bits;
                if (!gbr_get(rb, k + 1u, bits)) goto done;
                const uint32_t mag1 = lt < 4u ? lt : (1u << k) | (bits & ((1u << k) - 1u)), sign = bits >> k;
                if (mag1 + 1u > 5220u) goto done;
                R[nr++] = pos | sign << 12 | (mag1 + 1u) << 13;   /* scan position: k_idct maps it to the natural index */
                pos++;
            }
        }
        M[32] = nr;
        ok = x0 == 0 && x1 == 0 && gbr_finished(ra) && gbr_finished(rb);
    }
done:
    status[i] = ok;
}

/* the reference's 16-point inverse (VF_DCT16_INV_BODY), every operation rounded */
__device__ __forceinline__ void idct16(const float *in, float *out) {
    const float t1 = in[0], t2 = in[1], t3 = in[2], t4 = in[3], t5 = in[4], t6 = in[5], t7 = in[6], t8 = in[7], t9 = in[8], t10 = in[9],
                t11 = in[10], t12 = in[11], t13 = in[12], t14 = in[13], t15 = in[14], t16 = in[15];
    const float t17 = t4 + t2, t18 = t6 + t4, t19 = t8 + t6, t20 = t10 + t8, t21 = t12 + t10, t22 = t14 + t12, t23 = t16 + t14;
    const float t24 = t7 + t3, t25 = t11 + t7, t26 = t15 + t11, t27 = t13 + t5;
    const float t28 = t9 * 7.071067812e-01f, t29 = t1 + t28, t30 = t1 - t28;
    const float t31 = t27 * 7.071067812e-01f, t32 = t5 + t31, t33 = t5 - t31;
    const float t34 = t32 * 5.411961001e-01f, t35 = t29 + t34, t36 = t29 - t34;
    const float t37 = t33 * 1.306562965e+00f, t38 = t30 + t37, t39 = t30 - t37;
    const float t40 = t26 + t24, t41 = t25 * 7.071067812e-01f, t42 = t3 + t41, t43 = t3 - t41;
    const float t44 = t40 * 7.071067812e-01f, t45 = t24 + t44, t46 = t24 - t44;
    const float t47 = t45 * 5.411961001e-01f, t48 = t42 + t47, t49 = t42 - t47;
    const float t50 = t46 * 1.306562965e+00f, t51 = t43 + t50, t52 = t43 - t50;
    const float t53 = t48 * 5.097955791e-01f, t54 = t35 + t53, t55 = t35 - t53;
    const float t56 = t51 * 6.013448869e-01f, t57 = t38 + t56, t58 = t38 - t56;
    const float t59 = t52 * 8.999762231e-01f, t60 = t39 + t59, t61 = t39 - t59;
    const float t62 = t49 * 2.562915448e+00f, t63 = t36 + t62, t64 = t36 - t62;
    const float t65 = t19 + t17, t66 = t21 + t19, t67 = t23 + t21, t68 = t22 + t18;
    const float t69 = t20 * 7.071067812e-01f, t70 = t2 + t69, t71 = t2 - t69;
    const float t72 = t68 * 7.071067812e-01f, t73 = t18 + t72, t74 = t18 - t72;
    const float t75 = t73 * 5.411961001e-01f, t76 = t70 + t75, t77 = t70 - t75;
    const float t78 = t74 * 1.306562965e+00f, t79 = t71 + t78, t80 = t71 - t78;
    const float t81 = t67 + t65, t82 = t66 * 7.071067812e-01f, t83 = t17 + t82, t84 = t17 - t82;
    const float t85 = t81 * 7.071067812e-01f, t86 = t65 + t85, t87 = t65 - t85;
    const float t88 = t86 * 5.411961001e-01f, t89 = t83 + t88, t90 = t83 - t88;
    const float t91 = t87 * 1.306562965e+00f, t92 = t84 + t91, t93 = t84 - t91;
    const float t94 = t89 * 5.097955791e-01f, t95 = t76 + t94, t96 = t76 - t94;
    const float t97 = t92 * 6.013448869e-01f, t98 = t79 + t97, t99 = t79 - t97;
    const float t100 = t93 * 8.999762231e-01f, t101 = t80 + t100, t102 = t80 - t100;
    const float t103 = t90 * 2.562915448e+00f, t104 = t77 + t103, t105 = t77 - t103;
    const float t106 = t95 * 5.024192862e-01f, t107 = t54 + t106, t108 = t54 - t106;
    const float t109 = t98 * 5.224986149e-01f, t110 = t57 + t109, t111 = t57 - t109;
    const float t112 = t101 * 5.669440348e-01f, t113 = t60 + t112, t114 = t60 - t112;
    const float t115 = t104 * 6.468217834e-01f, t116 = t63 + t115, t117 = t63 - t115;
    const float t118 = t105 * 7.881546235e-01f, t119 = t64 + t118, t120 = t64 - t118;
    const float t121 = t102 * 1.060677686e+00f, t122 = t61 + t121, t123 = t61 - t121;
    const float t124 = t99 * 1.722447098e+00f, t125 = t58 + t124, t126 = t58 - t124;
    const float t127 = t96 * 5.101148619e+00f, t128 = t55 + t127, t129 = t55 - t127;
    out[0] = t107; out[1] = t110; out[2] = t113; out[3] = t116; out[4] = t119; out[5] = t122; out[6] = t125; out[7] = t128;
    out[8] = t129; out[9] = t126; out[10] = t123; out[11] = t120; out[12] = t117; out[13] = t114; out[14] = t111; out[15] = t108;
}
__device__ __forceinline__ uint32_t round8(float v) {   /* vf_scatter_block / vf_flat_value */
    v = v + 128.5f;
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    return (uint32_t)(int32_t)v;
}

/* shared block: (z, y) rows of 17 floats (bank-conflict free in all three passes) */
#define SI(z, y, x) (((z) * 16 + (y)) * 17 + (x))
__global__ void __launch_bounds__(256) k_idct(const gd_item *__restrict__ items, const uint8_t *__restrict__ in, const uint32_t *__restrict__ pos_tab, const uint32_t *__restrict__ rec,
                                              const uint32_t *__restrict__ meta, const uint32_t *__restrict__ status) {
    __shared__ float S[16 * 16 * 17];
    const int i = blockIdx.x >> 4, j = blockIdx.x & 15, t = threadIdx.x;
    if (!status[i]) return;
    const gd_item it = items[i];
    const uint32_t *M = meta + (size_t)i * META;
    const float *step = (const float *)(in + it.tab_off + TABW * 4);
    const int32_t dc = (int32_t)M[j];
    const uint32_t r0 = M[16 + j], r1 = M[16 + j + 1];
    float b0 = 0.0f;
    if (dc != 0) { const float mag = (float)(uint32_t)(dc < 0 ? -dc : dc) * step[0] * 0.015625f; b0 = dc < 0 ? -mag : mag; }
    /* output row (z, y) of block j: compact substream layout z (16) x y (32) x x (128) */
    uint8_t *orow = (uint8_t *)it.dst + (size_t)(t >> 4) * it.zs + ((j >> 3) * 16 + (t & 15)) * 128 + (j & 7) * 16;
    if (r0 == r1) {   /* DC only */
        const uint32_t v = round8(b0) * 0x01010101u;
        *(uint4 *)orow = make_uint4(v, v, v, v);
        return;
    }
    for (int k = t; k < 16 * 16 * 17; k += 256) S[k] = 0.0f;
    __syncthreads();
    if (t == 0) S[0] = b0;
    for (uint32_t r = r0 + t; r < r1; r += 256) {
        const uint32_t w = __ldg(rec + it.rec_off + r), nat = __ldg(pos_tab + (w & 4095u)) & 4095u, sign = w >> 12 & 1u, mag = w >> 13;
        const uint32_t x = nat & 15u, y = nat >> 4 & 15u, z = nat >> 8, z0 = (x == 0) + (y == 0) + (z == 0);
        const float v = ((float)mag + (mag == 1u ? 0.15f : 0.30f)) * step[x + y + z] * c_scale[z0];
        S[SI(z, y, x)] = sign ? -v : v;
    }
    __syncthreads();
    float a[16], o[16];
    { const int z = t >> 4, y = t & 15; for (int k = 0; k < 16; k++) a[k] = S[SI(z, y, k)]; idct16(a, o); for (int k = 0; k < 16; k++) S[SI(z, y, k)] = o[k]; }
    __syncthreads();
    { const int z = t >> 4, x = t & 15; for (int k = 0; k < 16; k++) a[k] = S[SI(z, k, x)]; idct16(a, o); for (int k = 0; k < 16; k++) S[SI(z, k, x)] = o[k]; }
    __syncthreads();
    { const int y = t >> 4, x = t & 15; for (int k = 0; k < 16; k++) a[k] = S[SI(k, y, x)]; idct16(a, o); for (int k = 0; k < 16; k++) S[SI(k, y, x)] = o[k]; }
    __syncthreads();
    { const int z = t >> 4, y = t & 15; uint32_t w[4];
      for (int q = 0; q < 4; q++) w[q] = round8(S[SI(z, y, 4 * q)]) | round8(S[SI(z, y, 4 * q + 1)]) << 8 | round8(S[SI(z, y, 4 * q + 2)]) << 16 | round8(S[SI(z, y, 4 * q + 3)]) << 24;
      *(uint4 *)orow = make_uint4(w[0], w[1], w[2], w[3]); }
}

/* ---------------- host: device state, batching service ---------------- */
enum { NSLOT = 3, MAXREQ = 4096 };
enum { S_FREE, S_OPEN, S_GPU, S_DRAIN };
struct gd_req { int slot; uint32_t item0, nitems, mask, tab_off, out0; uint8_t *dst, *ddst; int state; };   /* ddst: device chunk (no copy out) */   /* state: 0 pending, 1 ok, 2 GPU decode failed, 3 GPU unavailable */
struct gd_slot {
    int state, writers, readers, nreq;
    uint8_t *h_in, *h_out; gd_item *h_items; gd_group *h_groups; uint32_t *h_status;
    size_t in_used, rec_used; uint32_t nitems, nout;
    gd_req *req[MAXREQ];
};
static struct {
    pthread_once_t once; pthread_mutex_t mu; pthread_cond_t cv; pthread_t th;
    int ok, broken, dev;
    size_t in_cap, rec_cap; uint32_t max_items, min_bytes;
    gd_slot slot[NSLOT]; int open;
    uint8_t *d_in, *d_out; gd_item *d_items; gd_group *d_groups; uint32_t *d_rec, *d_meta, *d_status, *d_pos;
    cudaStream_t st; cudaEvent_t ev;
    uint16_t scan[BLKV]; int have_scan;
    uint64_t n_subs, n_batches, n_redo, n_busy, n_cpu_busy; double gpu_s;
    double max_s, cool_s; int64_t busy_us;   /* a GPU shared with saturating kernels: round trips over max_s -> the CPU for cool_s */
} G = {PTHREAD_ONCE_INIT, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define GD_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "gpudec: %s (%s:%d), decoding on the CPU from now on\n", cudaGetErrorString(e_), __FILE__, __LINE__); goto fail; } } while (0)

static int take_free(void) {
    for (int k = 0; k < NSLOT; k++) if (G.slot[k].state == S_FREE) { gd_slot *S = &G.slot[k]; S->state = S_OPEN; S->in_used = S->rec_used = 0; S->nitems = S->nout = 0; S->nreq = 0; S->writers = S->readers = 0; return k; }
    return -1;
}

/* one batch, synchronously on the service thread; returns 0, or -1 on a device error */
static int run_batch(gd_slot *S) {
    const double t0 = now_s(); const uint32_t n = S->nitems;
    GD_CK(cudaMemcpyAsync(G.d_in, S->h_in, S->in_used, cudaMemcpyHostToDevice, G.st));
    for (int r = 0; r < S->nreq; r++) S->h_groups[r] = (gd_group){S->req[r]->item0, S->req[r]->nitems, S->req[r]->tab_off, 0};
    GD_CK(cudaMemcpyAsync(G.d_items, S->h_items, n * sizeof(gd_item), cudaMemcpyHostToDevice, G.st));
    GD_CK(cudaMemcpyAsync(G.d_groups, S->h_groups, S->nreq * sizeof(gd_group), cudaMemcpyHostToDevice, G.st));
    k_entropy<<<S->nreq, 32, 0, G.st>>>(G.d_groups, G.d_items, G.d_in, G.d_pos, G.d_rec, G.d_meta, G.d_status);
    GD_CK(cudaGetLastError());
    k_idct<<<n * 16, 256, 0, G.st>>>(G.d_items, G.d_in, G.d_pos, G.d_rec, G.d_meta, G.d_status);
    GD_CK(cudaGetLastError());
    GD_CK(cudaMemcpyAsync(S->h_status, G.d_status, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.st));
    if (S->nout) GD_CK(cudaMemcpyAsync(S->h_out, G.d_out, (size_t)S->nout * SUB_BYTES, cudaMemcpyDeviceToHost, G.st));
    GD_CK(cudaEventRecord(G.ev, G.st));
    GD_CK(cudaEventSynchronize(G.ev));
    { const double dt = now_s() - t0; G.gpu_s += dt;
      if (dt > G.max_s) { __atomic_store_n(&G.busy_us, (int64_t)((now_s() + G.cool_s) * 1e6), __ATOMIC_RELAXED); G.n_busy++; } }   /* starved (time sliced next to another job's kernels): the CPU decodes for a while */
    return 0;
fail:
    return -1;
}

static void *service(void *arg) {
    (void)arg;
    if (cudaSetDevice(G.dev) != cudaSuccess) { pthread_mutex_lock(&G.mu); G.broken = 1; pthread_cond_broadcast(&G.cv); pthread_mutex_unlock(&G.mu); return nullptr; }
    pthread_mutex_lock(&G.mu);
    for (;;) {
        while (!G.broken && !(G.open >= 0 && G.slot[G.open].nitems > 0 && G.slot[G.open].writers == 0)) pthread_cond_wait(&G.cv, &G.mu);
        if (G.broken) break;
        const int k = G.open; gd_slot *S = &G.slot[k];
        S->state = S_GPU; G.open = take_free();
        pthread_cond_broadcast(&G.cv);
        pthread_mutex_unlock(&G.mu);
        const int rc = run_batch(S);
        pthread_mutex_lock(&G.mu);
        if (rc) G.broken = 1;
        G.n_batches++;
        int readers = 0;
        for (int r = 0; r < S->nreq; r++) {
            gd_req *q = S->req[r]; int bad = 0;
            if (!rc) for (uint32_t u = 0; u < q->nitems; u++) bad |= !S->h_status[q->item0 + u];
            q->state = rc ? 3 : bad ? 2 : 1;
            readers += q->state == 1 && !q->ddst;
            if (q->state == 1) G.n_subs += q->nitems; else if (q->state == 2) G.n_redo++;
        }
        S->readers = readers; S->state = readers ? S_DRAIN : S_FREE;
        if (G.open < 0) G.open = take_free();
        pthread_cond_broadcast(&G.cv);
        if (G.broken) break;
    }
    /* broken: fail every queued request, close the open slot */
    for (int k = 0; k < NSLOT; k++) {
        gd_slot *S = &G.slot[k];
        if (S->state == S_OPEN) for (int r = 0; r < S->nreq; r++) S->req[r]->state = 3;
    }
    pthread_cond_broadcast(&G.cv);
    pthread_mutex_unlock(&G.mu);
    return nullptr;
}

static void gd_stats_atexit(void) {
    fprintf(stderr, "gpudec: %llu substreams on the GPU in %llu batches (%.0f / batch), %llu requests redone on the CPU, GPU round trips %.2f s\n",
            (unsigned long long)G.n_subs, (unsigned long long)G.n_batches, G.n_batches ? (double)G.n_subs / G.n_batches : 0.0, (unsigned long long)G.n_redo, G.gpu_s);
    if (G.n_busy) fprintf(stderr, "gpudec: %llu round trips over %.0f ms (GPU busy): %llu requests decoded on the CPU meanwhile\n", (unsigned long long)G.n_busy, G.max_s * 1e3, (unsigned long long)G.n_cpu_busy);
}
static void gd_init(void) {
    const char *e = getenv("UFSM_GPU_DECODE");
    if (!e || !*e || !strcmp(e, "0")) return;
    G.dev = getenv("UFSM_GPU_DECODE_DEV") ? atoi(getenv("UFSM_GPU_DECODE_DEV")) : 0;
    const char *mi = getenv("UFSM_GPU_DECODE_ITEMS");
    G.max_items = mi && atoi(mi) > 0 ? (uint32_t)atoi(mi) : 1024u;
    const char *mb = getenv("UFSM_GPU_DECODE_MIN_BYTES");
    G.min_bytes = mb ? (uint32_t)atoi(mb) : 2048u;   /* per substream, compressed (Paris 4 CT q 2: ~9 KB; ink q 32: ~200 B) */
    const char *ms = getenv("UFSM_GPU_DECODE_MAX_MS"), *cs = getenv("UFSM_GPU_DECODE_COOL_S");
    G.max_s = (ms && atof(ms) > 0 ? atof(ms) : 60.0) * 1e-3;   /* an idle GPU: 5-12 ms for a full batch */
    G.cool_s = cs && atof(cs) >= 0 ? atof(cs) : 2.0;
    G.in_cap = (size_t)G.max_items * 40000 + ((size_t)16 << 20);   /* q 2 substreams are ~10-20 KB */
    G.rec_cap = (size_t)G.max_items * 49152 + ((size_t)16 << 20);   /* records (u32); a substream reserves its bound, 8 x its bypass bytes (q 2: ~5 KB) */
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || G.dev < 0 || G.dev >= nd) { cudaGetLastError(); fprintf(stderr, "gpudec: no CUDA device %d, decoding on the CPU\n", G.dev); return; }
    int prev = 0; cudaGetDevice(&prev);
    GD_CK(cudaSetDevice(G.dev));
    GD_CK(cudaStreamCreateWithFlags(&G.st, cudaStreamNonBlocking));
    GD_CK(cudaEventCreateWithFlags(&G.ev, cudaEventBlockingSync | cudaEventDisableTiming));   /* the service thread sleeps, it does not spin */
    GD_CK(cudaMalloc(&G.d_in, G.in_cap));
    GD_CK(cudaMalloc(&G.d_out, (size_t)G.max_items * SUB_BYTES));
    GD_CK(cudaMalloc(&G.d_items, G.max_items * sizeof(gd_item)));
    GD_CK(cudaMalloc(&G.d_groups, MAXREQ * sizeof(gd_group)));
    GD_CK(cudaMalloc(&G.d_rec, G.rec_cap * 4));
    GD_CK(cudaMalloc(&G.d_meta, (size_t)G.max_items * META * 4));
    GD_CK(cudaMalloc(&G.d_status, G.max_items * 4));
    GD_CK(cudaMalloc(&G.d_pos, BLKV * 4));
    for (int k = 0; k < NSLOT; k++) {
        gd_slot *S = &G.slot[k];
        GD_CK(cudaHostAlloc(&S->h_in, G.in_cap, cudaHostAllocPortable));
        GD_CK(cudaHostAlloc(&S->h_out, (size_t)G.max_items * SUB_BYTES, cudaHostAllocPortable));
        GD_CK(cudaHostAlloc(&S->h_items, G.max_items * sizeof(gd_item), cudaHostAllocPortable));
        GD_CK(cudaHostAlloc(&S->h_groups, MAXREQ * sizeof(gd_group), cudaHostAllocPortable));
        GD_CK(cudaHostAlloc(&S->h_status, G.max_items * 4, cudaHostAllocPortable));
        S->state = S_FREE;
    }
    cudaSetDevice(prev);
    G.open = take_free();
    G.ok = 1;
    if (getenv("UFSM_GPU_DECODE_STATS") && strcmp(getenv("UFSM_GPU_DECODE_STATS"), "0")) atexit(gd_stats_atexit);
    return;
fail:
    cudaGetLastError(); cudaSetDevice(prev);
}

/* the scan-position table goes to the device with the first request (volcomp.h's VF_SCAN16 lives on the C side) */
static int pos_upload(const uint16_t *scan) {
    uint32_t pt[BLKV];
    for (uint32_t pos = 0; pos < BLKV; pos++) {
        const uint32_t nat = scan[pos], x = nat & 15u, y = nat >> 4 & 15u, z = nat >> 8;
        const uint32_t z0 = (x == 0) + (y == 0) + (z == 0), band = pos < 128u ? 0u : pos < 1024u ? 1u : 2u;
        pt[pos] = nat | (x + y + z) << 12 | z0 << 18 | band << 20;
    }
    int prev = 0; cudaGetDevice(&prev);
    const int bad = cudaSetDevice(G.dev) != cudaSuccess || cudaMemcpy(G.d_pos, pt, sizeof pt, cudaMemcpyHostToDevice) != cudaSuccess;
    cudaSetDevice(prev);
    return bad ? -1 : 0;
}

extern "C" int gpudec_enabled(void) { pthread_once(&G.once, gd_init); return G.ok && !G.broken; }
extern "C" int gpudec_threads(int nthreads) {
    if (!gpudec_enabled()) return nthreads;
    const char *e = getenv("UFSM_GPU_DECODE_THREADS");
    const int t = e && atoi(e) > 0 ? atoi(e) : 32;
    return t > nthreads ? t : nthreads;
}
extern "C" void gpudec_stats(uint64_t *subs, uint64_t *batches, uint64_t *redo, double *gpu_s) {
    pthread_mutex_lock(&G.mu);
    if (subs) *subs = G.n_subs;
    if (batches) *batches = G.n_batches;
    if (redo) *redo = G.n_redo;
    if (gpu_s) *gpu_s = G.gpu_s;
    pthread_mutex_unlock(&G.mu);
}

static int gd_decode(const gd_chunk *c, uint32_t mask, uint8_t *dst, uint8_t *ddst) {
    if (!mask) return 0;
    if (!gpudec_enabled()) return -1;
    if ((double)__atomic_load_n(&G.busy_us, __ATOMIC_RELAXED) > now_s() * 1e6) { __atomic_fetch_add(&G.n_cpu_busy, 1, __ATOMIC_RELAXED); return -1; }   /* after a starved round trip */
    const uint32_t k = (uint32_t)__builtin_popcount(mask);
    size_t need_in = TAB_BYTES, need_rec = 0, payload = 0;
    for (uint32_t s = 0; s < NSUB; s++) if (mask >> s & 1u) {
        need_in += ((c->tok_n[s] + 15u) & ~15u) + 16 + ((c->byp_n[s] + 15u) & ~15u) + 16; need_rec += (size_t)c->byp_n[s] * 8u + 1u;
        payload += (size_t)c->tok_n[s] + c->byp_n[s];
    }
    if (need_in > G.in_cap || need_rec > G.rec_cap || k > G.max_items) return -1;
    if (payload < (size_t)k * G.min_bytes) return -1;   /* near-empty substreams (flat blocks: ink, air) decode faster on the CPU than a round trip */
    gd_req q = {}; q.mask = mask; q.dst = dst; q.ddst = ddst; q.nitems = k;
    const uint32_t kout = ddst ? 0 : k;
    pthread_mutex_lock(&G.mu);
    if (!G.have_scan) {
        if (pos_upload(c->scan)) { G.broken = 1; pthread_cond_broadcast(&G.cv); pthread_mutex_unlock(&G.mu); return -1; }
        G.have_scan = 1;
        if (pthread_create(&G.th, nullptr, service, nullptr)) { G.broken = 1; pthread_mutex_unlock(&G.mu); return -1; }
        pthread_detach(G.th);
    }
    gd_slot *S = nullptr;
    for (;;) {
        if (G.broken) { pthread_mutex_unlock(&G.mu); return -1; }
        if (G.open >= 0) {
            S = &G.slot[G.open];
            if (S->in_used + need_in <= G.in_cap && S->rec_used + need_rec <= G.rec_cap && S->nitems + k <= G.max_items && S->nout + kout <= G.max_items && S->nreq < MAXREQ) break;
        }
        pthread_cond_wait(&G.cv, &G.mu);
    }
    q.slot = G.open; q.item0 = S->nitems; q.out0 = S->nout; S->nout += kout;
    size_t off = S->in_used, roff = S->rec_used;
    S->in_used += need_in; S->rec_used += need_rec; S->nitems += k; S->req[S->nreq++] = &q; S->writers++;
    pthread_mutex_unlock(&G.mu);

    /* pack: tables + steps, then each substream's two streams (16-byte aligned, 16 bytes of slack for the word reader) */
    const uint32_t tab_off = (uint32_t)off; q.tab_off = tab_off;
    for (int m = 0; m < 10; m++) memcpy(S->h_in + off + (size_t)m * 4096, c->dt[m], 4096);
    memcpy(S->h_in + off + TABW * 4, c->step, NRADII * 4);
    off += TAB_BYTES;
    uint32_t u = 0;
    for (uint32_t s = 0; s < NSUB; s++) if (mask >> s & 1u) {
        gd_item *it = &S->h_items[q.item0 + u++];
        const uint8_t *p = c->payload + c->off[s];
        it->tab_off = tab_off;
        it->tok_off = (uint32_t)off; it->tok_n = c->tok_n[s]; memcpy(S->h_in + off, p, c->tok_n[s]); off += ((c->tok_n[s] + 15u) & ~15u) + 16;
        it->byp_off = (uint32_t)off; it->byp_n = c->byp_n[s]; memcpy(S->h_in + off, p + c->tok_n[s], c->byp_n[s]); off += ((c->byp_n[s] + 15u) & ~15u) + 16;
        it->rec_off = (uint32_t)roff; roff += (size_t)c->byp_n[s] * 8u + 1u;
        if (ddst) { it->dst = (uint64_t)(uintptr_t)(ddst + ((size_t)16u * (s >> 2) * 128u + 32u * (s & 3u)) * 128u); it->zs = 128u * 128u; }
        else { it->dst = (uint64_t)(uintptr_t)(G.d_out + (size_t)(q.out0 + u - 1) * SUB_BYTES); it->zs = 32u * 128u; }
    }

    pthread_mutex_lock(&G.mu);
    S->writers--;
    pthread_cond_broadcast(&G.cv);
    while (q.state == 0) pthread_cond_wait(&G.cv, &G.mu);
    pthread_mutex_unlock(&G.mu);
    if (q.state != 1) return q.state == 2 ? 1 : -1;
    if (ddst) return 0;
    u = 0;
    for (uint32_t s = 0; s < NSUB; s++) if (mask >> s & 1u) {   /* compact substream -> its 16 z planes of 32 x 128 in the chunk */
        const uint8_t *src = S->h_out + (size_t)(q.out0 + u++) * SUB_BYTES;
        for (uint32_t z = 0; z < 16; z++) memcpy(dst + ((size_t)(16u * (s >> 2) + z) * 128u + 32u * (s & 3u)) * 128u, src + (size_t)z * 4096u, 4096);
    }
    pthread_mutex_lock(&G.mu);
    if (--S->readers == 0) { S->state = S_FREE; if (G.open < 0) G.open = take_free(); pthread_cond_broadcast(&G.cv); }
    pthread_mutex_unlock(&G.mu);
    return 0;
}
extern "C" int gpudec_decode(const gd_chunk *c, uint32_t mask, uint8_t *dst) { return gd_decode(c, mask, dst, nullptr); }
extern "C" int gpudec_decode_dev(const gd_chunk *c, uint32_t mask, uint8_t *dchunk) {
    if (!dchunk || !gpudec_enabled()) return -1;
    cudaPointerAttributes a;   /* device memory of the decoder's device only */
    if (cudaPointerGetAttributes(&a, dchunk) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (a.type != cudaMemoryTypeDevice || a.device != G.dev) return -1;
    return gd_decode(c, mask, nullptr, dchunk);
}
