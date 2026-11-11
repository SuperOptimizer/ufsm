/* src/band_onfly.h: the pair-rule winding phase computed from the codes on read. */
#include "band_onfly.h"
#include "band.h"
#include "bandgpu.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float rmax; int halo; } onfly_ctx;
static int g_gpu = -1, g_slots = 2, g_budget_mb = 1024;
static _Atomic uint64_t g_reads, g_tiles, g_skipped, g_vout, g_vtile, g_ns_pair, g_ns_read;
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
    g_gpu = gpu; g_slots = slots < 1 ? 1 : slots; g_budget_mb = budget_mb < 64 ? 64 : budget_mb;
    if (gpu < 0) return 0;
    if (!band_pair_gpu) return -1;   /* (a binary without nn_band.o) */
    band_pair_gpu_slots(g_slots, gpu);
    if (reserve && band_pair_gpu_reserve && band_pair_gpu_reserve(phase_onfly_tile_edge(0))) return -1;
    return 0;
}
void phase_onfly_get_stats(phase_onfly_stats *st) {
    st->reads = g_reads; st->tiles = g_tiles; st->skipped = g_skipped; st->vox_out = g_vout; st->vox_tile = g_vtile;
    st->s_pair = 1e-9 * (double)g_ns_pair; st->s_read = 1e-9 * (double)g_ns_read;
}

static int pair_tile(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase) {
    if (g_gpu >= 0 && band_pair_gpu && !band_pair_gpu(codes, n, rmax, phase)) return 0;
    return band_field_phase_pair(codes, n, rmax, phase, nullptr);
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
    uint8_t *cb = malloc(cap), *pb = malloc(cap), *rb = malloc(cap);
    if (!cb || !pb || !rb) { free(cb); free(pb); free(rb); return -1; }
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
    free(cb); free(pb); free(rb);
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
