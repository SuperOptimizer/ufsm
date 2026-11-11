/* src/band_onfly.h: the pair-rule winding phase computed from the codes on read. */
#include "band_onfly.h"
#include "band.h"
#include "bandgpu.h"
#include "volcomp.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float rmax; int halo; } onfly_ctx;
static int g_gpu = -1, g_slots = 2, g_budget_mb = 1024;
static int g_cpu_decode;   /* UFSM_ONFLY_CPU_DECODE=1: the codes decoded on the CPU (A/B) */
static _Atomic uint64_t g_gputiles, g_reads, g_tiles, g_skipped, g_vout, g_vtile, g_ns_pair, g_ns_read;
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* device bytes of one tile of edge E (nn_band.cu bg_run: codes + phase + two int32 feature arrays, the envelope scratch of 4 int32 /
   float arrays over (L + 1) x the chunk's lines: ~4 B per voxel, nn_band.cu bg_lines) */
static double tile_bytes(double E) { double lines = E * E * E / (4 * (E + 1)); if (lines < 32768) lines = 32768; if (lines > E * E) lines = E * E; return E * E * E * 10 + 16 * (E + 1) * lines; }
int phase_onfly_tile_edge(float rmax) {
    const int h = (int)ceilf(rmax) + 2; int E = 2 * h + 32;
    while (tile_bytes(E + 8) <= (double)g_budget_mb * (1 << 20)) E += 8;
    return E;
}
int phase_onfly_config(int gpu, int slots, int budget_mb, int reserve) {
    g_cpu_decode = getenv("UFSM_ONFLY_CPU_DECODE") && atoi(getenv("UFSM_ONFLY_CPU_DECODE"));
    g_gpu = gpu; g_slots = slots < 1 ? 1 : slots; g_budget_mb = budget_mb < 64 ? 64 : budget_mb;
    if (gpu < 0) return 0;
    if (!band_pair_gpu) return -1;   /* (a binary without nn_band.o) */
    band_pair_gpu_slots(g_slots, gpu);
    if (reserve && band_pair_gpu_reserve && band_pair_gpu_reserve(phase_onfly_tile_edge(0))) return -1;
    return 0;
}
void phase_onfly_get_stats(phase_onfly_stats *st) {
    st->reads = g_reads; st->tiles = g_tiles; st->skipped = g_skipped; st->vox_out = g_vout; st->vox_tile = g_vtile;
    st->gpu_tiles = g_gputiles; st->s_pair = 1e-9 * (double)g_ns_pair; st->s_read = 1e-9 * (double)g_ns_read;
}

static int pair_tile(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase) {
    if (g_gpu >= 0 && band_pair_gpu && !band_pair_gpu(codes, n, rmax, phase)) return 0;
    return band_field_phase_pair(codes, n, rmax, phase, nullptr);
}

/* tile buffer sets, a pool of --onfly-slots shared by the reading threads (each read takes one for its tiles), kept between reads
   (2026-11-11: fresh 1.2 GB mallocs per window page-faulted ~1 s on the pod) and pinned when the pair rule runs on the GPU (the codes up
   and the phase down at the PCIe rate: the pageable D2H of a 1064^3 tile took 0.9-1.1 s) */
typedef struct { uint8_t *cb, *pb, *rb; size_t cap; int pinned, busy; } onfly_buf;
enum { OB_MAX = 16 };
static onfly_buf g_ob[OB_MAX]; static pthread_mutex_t g_obmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_obcv = PTHREAD_COND_INITIALIZER;
static void ob_release(onfly_buf *B) { if (B->pinned) { band_pair_gpu_host_free(B->cb); band_pair_gpu_host_free(B->pb); } else { free(B->cb); free(B->pb); } free(B->rb); B->cb = B->pb = B->rb = nullptr; B->cap = 0; }
static onfly_buf *ob_take(size_t cap) {
    const int n = g_slots < OB_MAX ? g_slots : OB_MAX; onfly_buf *B = nullptr;
    pthread_mutex_lock(&g_obmu);
    for (;;) { for (int k = 0; k < n; k++) if (!g_ob[k].busy) { B = &g_ob[k]; break; } if (B) break; pthread_cond_wait(&g_obcv, &g_obmu); }
    B->busy = 1; pthread_mutex_unlock(&g_obmu);
    if (cap > B->cap) {
        ob_release(B);
        B->pinned = g_gpu >= 0 && band_pair_gpu_host_alloc && band_pair_gpu_host_free;
        if (B->pinned) { B->cb = band_pair_gpu_host_alloc(cap); B->pb = band_pair_gpu_host_alloc(cap); }
        if (!B->pinned || !B->cb || !B->pb) { if (B->pinned) { band_pair_gpu_host_free(B->cb); band_pair_gpu_host_free(B->pb); } B->pinned = 0; B->cb = malloc(cap); B->pb = malloc(cap); }
        B->rb = malloc(cap);
        if (!B->cb || !B->pb || !B->rb) { ob_release(B); pthread_mutex_lock(&g_obmu); B->busy = 0; pthread_cond_signal(&g_obcv); pthread_mutex_unlock(&g_obmu); return nullptr; }
        B->cap = cap;
    }
    return B;
}
static void ob_give(onfly_buf *B) { pthread_mutex_lock(&g_obmu); B->busy = 0; pthread_cond_signal(&g_obcv); pthread_mutex_unlock(&g_obmu); }


/* the GPU path for one tile: the codes chunks over the clipped box (ro, rn) fetched raw (z3_chunk_raw: the CPU only reads), parsed and
   packed (tables, aligned substreams) for band_pair_gpu_codes, which decodes them on the device, runs the pair rule and copies the crop
   (the tile interior, at h .. h + tn of the box) down into out. 0 done; 2 no labelled chunk (the tile stays unknown); 1 / -1 / -2: the
   CPU path (a chunk the GPU did not decode, a CUDA failure, codes not in 128^3 volcomp lossless chunks). */
typedef struct { uint8_t *b; size_t n, cap; bg_lchunk *ch; int nch, ccap; } onfly_pack;
static thread_local onfly_pack t_pk;
static size_t pk_put(onfly_pack *K, const void *src, size_t n, size_t align) {
    size_t o = (K->n + align - 1) & ~(align - 1);
    if (o + n + 64 > K->cap) { size_t c = K->cap ? K->cap : 1 << 20; while (c < o + n + 64) c *= 2; uint8_t *nb = realloc(K->b, c); if (!nb) return SIZE_MAX; K->b = nb; K->cap = c; }
    if (src) memcpy(K->b + o, src, n);
    K->n = o + n; return o;
}
static int tile_gpu(z3 *codes, const int64_t co[3], const int64_t cn[3], const int64_t ro[3], const int64_t rn[3], int h, const int64_t tn[3], float rmax, uint8_t *out) {
    const z3_meta *m = z3_meta_of(codes);
    if (m->chunk[0] != 128 || m->chunk[1] != 128 || m->chunk[2] != 128 || m->esize != 1 || m->q != 0) return -2;
    onfly_pack *K = &t_pk; K->n = 0; K->nch = 0;
    for (int64_t cz = ro[0] / 128; cz * 128 < ro[0] + rn[0]; cz++) for (int64_t cy = ro[1] / 128; cy * 128 < ro[1] + rn[1]; cy++) for (int64_t cx = ro[2] / 128; cx * 128 < ro[2] + rn[2]; cx++) {
        uint8_t *buf = nullptr; size_t len = 0; const int r = z3_chunk_raw(codes, cz, cy, cx, &buf, &len);
        if (r < 0) return -1;
        if (!r) continue;
        uint32_t mode; bg_lchunk C; memset(&C, 0, sizeof C); C.o[0] = (int32_t)(cz * 128 - co[0]); C.o[1] = (int32_t)(cy * 128 - co[1]); C.o[2] = (int32_t)(cx * 128 - co[2]);
        if (vll_chunk_mode(buf, len, &mode) != VOLCOMP_OK || (mode != VLL_MODE_CODED && mode != VLL_MODE_RAW && mode != VLL_MODE_CONST)) { free(buf); return -2; }
        C.mode = mode;
        if (mode == VLL_MODE_CONST) { if (len != VF_HDR_BYTES + 1u) { free(buf); return -2; } C.value = buf[VF_HDR_BYTES]; if (!C.value) { free(buf); continue; } }
        else if (mode == VLL_MODE_RAW) { if (len != VF_HDR_BYTES + (size_t)VOLCOMP_CHUNK_VOXELS) { free(buf); return -2; } if ((C.raw_off = (uint32_t)pk_put(K, buf + VF_HDR_BYTES, VOLCOMP_CHUNK_VOXELS, 16)) == UINT32_MAX) { free(buf); return -1; } }
        else {
            static thread_local vll_parsed lp;
            if (vll_parse(buf, len, &lp) != VOLCOMP_OK) { free(buf); return -2; }
            const size_t to = pk_put(K, nullptr, 8 * 1024 * 4, 16); if (to == SIZE_MAX) { free(buf); return -1; }
            uint32_t *T = (uint32_t *)(K->b + to);
            for (int q = 0; q < 8; q++) for (int i = 0; i < 1024; i++) { const vf_dentry e = lp.models[q].dt[i]; T[q * 1024 + i] = e.ns | (uint32_t)e.sym << 16 | (uint32_t)e.nb << 24; }
            C.tab_off = (uint32_t)to;
            for (int q = 0; q < 32; q++) {
                const size_t a = pk_put(K, lp.payload + lp.off[q], lp.tok_n[q], 4), b = pk_put(K, lp.payload + lp.off[q] + lp.tok_n[q], lp.byp_n[q], 4);
                if (a == SIZE_MAX || b == SIZE_MAX) { free(buf); return -1; }
                C.tok_off[q] = (uint32_t)a; C.tok_n[q] = lp.tok_n[q]; C.byp_off[q] = (uint32_t)b; C.byp_n[q] = lp.byp_n[q];
            }
        }
        free(buf);
        if (K->nch == K->ccap) { const int c = K->ccap ? 2 * K->ccap : 256; bg_lchunk *nc = realloc(K->ch, (size_t)c * sizeof *nc); if (!nc) return -1; K->ch = nc; K->ccap = c; }
        K->ch[K->nch++] = C;
    }
    if (K->n > UINT32_MAX - 64) return -2;
    if (!K->nch) return 2;
    const int n[3] = {(int)cn[0], (int)cn[1], (int)cn[2]}, lo[3] = {(int)(ro[0] - co[0]), (int)(ro[1] - co[1]), (int)(ro[2] - co[2])};
    const int hi[3] = {lo[0] + (int)rn[0], lo[1] + (int)rn[1], lo[2] + (int)rn[2]}, cro[3] = {h, h, h}, crn[3] = {(int)tn[0], (int)tn[1], (int)tn[2]};
    return band_pair_gpu_codes(K->ch, K->nch, K->b, K->n, n, lo, hi, rmax, cro, crn, out);
}

static int onfly_read(void *vctx, z3 *codes, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const onfly_ctx *C = vctx; const z3_meta *m = z3_meta_of(codes); const int h = C->halo;
    atomic_fetch_add(&g_reads, 1);
    const size_t N = (size_t)n[0] * n[1] * n[2]; memset(out, PHASE_UNKNOWN, N);
    int64_t lo[3], hi[3];   /* the part of the box inside the codes' shape (outside: the stored array's fill) */
    for (int d = 0; d < 3; d++) { lo[d] = o[d] < 0 ? 0 : o[d]; hi[d] = o[d] + n[d] < m->shape[d] ? o[d] + n[d] : m->shape[d]; if (hi[d] <= lo[d]) return 0; }
    const int E = phase_onfly_tile_edge(C->rmax), T = E - 2 * h;
    int cnt[3], ts[3];
    for (int d = 0; d < 3; d++) { const int64_t L = hi[d] - lo[d]; cnt[d] = (int)((L + T - 1) / T); ts[d] = (int)((L + cnt[d] - 1) / cnt[d]); }
    const size_t cap = (size_t)(ts[0] + 2 * h) * (ts[1] + 2 * h) * (ts[2] + 2 * h);
    onfly_buf *B = ob_take(cap); if (!B) return -1;
    uint8_t *cb = B->cb, *pb = B->pb, *rb = B->rb;
    int rc = 0;
    for (int a = 0; a < cnt[0] && !rc; a++) for (int b = 0; b < cnt[1] && !rc; b++) for (int c = 0; c < cnt[2] && !rc; c++) {
        const int ti[3] = {a, b, c}; int64_t to[3], tn[3], co[3], cn[3], ro[3], rn[3]; int ni[3];
        for (int d = 0; d < 3; d++) {
            to[d] = lo[d] + (int64_t)ti[d] * ts[d]; tn[d] = (to[d] + ts[d] < hi[d] ? to[d] + ts[d] : hi[d]) - to[d];
            co[d] = to[d] - h; cn[d] = tn[d] + 2 * h; ni[d] = (int)cn[d];
            ro[d] = co[d] < 0 ? 0 : co[d]; rn[d] = (co[d] + cn[d] < m->shape[d] ? co[d] + cn[d] : m->shape[d]) - ro[d];
        }
        const int any = z3_box_any(codes, ro, rn);
        if (any < 0) { rc = -1; break; }
        if (!any) { atomic_fetch_add(&g_skipped, 1); continue; }
        const double t0 = now_s();
        if (g_gpu >= 0 && band_pair_gpu_codes && !g_cpu_decode) {   /* the codes decoded on the device, only the tile interior comes down */
            const int g = tile_gpu(codes, co, cn, ro, rn, h, tn, C->rmax, pb);
            if (g == 2) { atomic_fetch_add(&g_skipped, 1); continue; }
            if (!g) {
                atomic_fetch_add(&g_ns_pair, (uint64_t)(1e9 * (now_s() - t0))); atomic_fetch_add(&g_tiles, 1); atomic_fetch_add(&g_vtile, (uint64_t)cn[0] * cn[1] * cn[2]); atomic_fetch_add(&g_gputiles, 1);
                for (int64_t z = 0; z < tn[0]; z++) for (int64_t y = 0; y < tn[1]; y++)
                    memcpy(out + ((size_t)(to[0] - o[0] + z) * n[1] + (to[1] - o[1] + y)) * n[2] + (to[2] - o[2]), pb + ((size_t)z * tn[1] + y) * tn[2], (size_t)tn[2]);
                continue;
            }
        }
        const size_t rnn = (size_t)rn[0] * rn[1] * rn[2], cnn = (size_t)cn[0] * cn[1] * cn[2];
        if (z3_read(codes, ro, rn, rb, nthreads)) { rc = -1; break; }
        size_t lab = 0; for (size_t k = 0; k < rnn; k++) lab += rb[k] > 0 && rb[k] < 255;
        if (!lab) { atomic_fetch_add(&g_skipped, 1); atomic_fetch_add(&g_ns_read, (uint64_t)(1e9 * (now_s() - t0))); continue; }
        memset(cb, 0, cnn);   /* (outside the shape: no codes, as phase-raster's zero-filled block) */
        for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
            memcpy(cb + ((size_t)(z + ro[0] - co[0]) * cn[1] + (y + ro[1] - co[1])) * cn[2] + (ro[2] - co[2]), rb + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
        const double t1 = now_s();
        if (pair_tile(cb, ni, C->rmax, pb)) { rc = -1; break; }
        atomic_fetch_add(&g_ns_read, (uint64_t)(1e9 * (t1 - t0))); atomic_fetch_add(&g_ns_pair, (uint64_t)(1e9 * (now_s() - t1)));
        atomic_fetch_add(&g_tiles, 1); atomic_fetch_add(&g_vtile, cnn);
        for (int64_t z = 0; z < tn[0]; z++) for (int64_t y = 0; y < tn[1]; y++)
            memcpy(out + ((size_t)(to[0] - o[0] + z) * n[1] + (to[1] - o[1] + y)) * n[2] + (to[2] - o[2]), pb + ((size_t)(z + h) * cn[1] + (y + h)) * cn[2] + h, (size_t)tn[2]);
    }
    atomic_fetch_add(&g_vout, N);
    ob_give(B);
    return rc;
}

z3 *phase_onfly_open(z3 *codes, float rmax) {
    if (!codes || !(rmax > 0)) return nullptr;
    onfly_ctx *C = malloc(sizeof *C); if (!C) return nullptr;
    C->rmax = rmax; C->halo = (int)ceilf(rmax) + 2;
    z3 *z = z3_open_derived(codes, PHASE_UNKNOWN, onfly_read, C, free);
    if (!z) free(C);
    return z;
}

int phase_onfly_decode_gpu(z3 *codes, const int64_t o[3], const int64_t n[3], uint8_t *out) {
    const z3_meta *m = z3_meta_of(codes); int64_t ro[3], rn[3], tn[3];
    for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; rn[d] = (o[d] + n[d] < m->shape[d] ? o[d] + n[d] : m->shape[d]) - ro[d]; if (rn[d] <= 0) return -2; tn[d] = n[d]; }
    if (!band_pair_gpu_codes) return -1;
    const int r = tile_gpu(codes, o, n, ro, rn, 0, tn, 0, out);
    if (r == 2) { memset(out, 0, (size_t)n[0] * n[1] * n[2]); return 0; }
    return r;
}
