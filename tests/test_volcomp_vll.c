/* volcomp's lossless encoder after the 2026-11-17 speedups (third_party/volcomp.h: word-at-a-time bit writer, table-driven block cost,
   DENSE tokens by table with interleaved histograms, the tANS lanes in registers; 2.0-2.5x on ink / CT chunks): byte for byte the
   encoder before them. Twelve synthetic 128^3 chunks covering every block and chunk mode (RAW chunk, CONST chunk, SPARSE, DENSE, RAW
   blocks, labels, ramps, half-empty) against the sizes and FNV-1a hashes the old encoder produced, each decoded back exactly; the lossy
   path (which shares the bit writer) round-trips at q 2 / 8 / 32 (its bytes were checked against the old encoder offline: ink, CT and
   these chunks, gcc; floating point makes them build-dependent, so no goldens here). The lossless sink decoder (src/volcomp_fast.h
   vllx_decode_sub_sink, its DENSE loop with the tANS lanes in registers, 2026-11-17) gives the chunk back, whole and in a partial box. */
#define VOLCOMP_IMPLEMENTATION
#include "volcomp.h"
#include "volcomp_fast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng = 88172645463325252ull;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static void synth(uint8_t *c, int k) {
    for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) {
        const int z = (int)(i >> 14), y = (int)(i >> 7 & 127), x = (int)(i & 127); uint32_t r = xr();
        switch (k) {
        case 0: c[i] = (uint8_t)r; break;                                          /* noise: a RAW chunk */
        case 1: c[i] = 77; break;                                                  /* CONST chunk */
        case 2: c[i] = (r % 997 == 0) ? (uint8_t)(r >> 16) : 0; break;             /* sparse */
        case 3: c[i] = (uint8_t)((x / 16 + y / 16 + z / 16) % 5 * 50); break;      /* labels: CONST blocks */
        case 4: c[i] = (uint8_t)(128 + (int)(r % 7) - 3); break;                   /* small noise: DENSE */
        case 5: c[i] = (uint8_t)(x * 2 + y + z); break;                            /* ramps */
        case 6: c[i] = (uint8_t)((r % 3 == 0) ? r >> 8 : 255); break;              /* heavy: RAW blocks */
        case 7: c[i] = (uint8_t)(z < 64 ? 0 : (uint8_t)(r % 40)); break;           /* half empty */
        case 8: c[i] = (uint8_t)((x ^ y ^ z) & 255); break;
        case 9: c[i] = (uint8_t)((r % 50 == 0) ? 255 : (r % 13 == 0 ? 1 : 0)); break;
        case 10: c[i] = (uint8_t)(i < 4096 ? r : 0); break;
        default: c[i] = (uint8_t)(((x - 64) * (x - 64) + (y - 64) * (y - 64) < 900) ? 200 + (int)(r % 30) : (int)(r % 4)); break;
        }
    }
}
static const struct { size_t n; uint64_t h; } gold[12] = {   /* the encoder before 2026-11-17 */
    {2097160, 0xc8d3981eea71f58dull},
    {9, 0xba592926bc4f43b8ull},
    {10819, 0x4b3076f5c20a6cbfull},
    {728, 0x6c1a61840e9acb09ull},
    {918300, 0x7830fb82f02da3adull},
    {6654, 0x9a50ad2051f8ce8cull},
    {1392273, 0xf157c6aad91d8d85ull},
    {808171, 0x57f2517a5f652b87ull},
    {592113, 0x867fd30eb762becfull},
    {231828, 0xb93a558e060016c9ull},
    {8970, 0x45ec91718b719eb6ull},
    {929947, 0x67174bece381bbe6ull},
};

int main(void) {
    uint8_t *c = malloc(VOLCOMP_CHUNK_VOXELS), *d = malloc(VOLCOMP_CHUNK_VOXELS), *o = malloc(VOLCOMP_ENCODE_BOUND); int fails = 0; double tl = 0;
    if (!c || !d || !o) return 1;
    for (int k = 0; k < 12; k++) {
        synth(c, k);
        size_t n = 0; struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        volcomp_status st = volcomp_encode(c, VOLCOMP_Q_LOSSLESS, o, VOLCOMP_ENCODE_BOUND, &n);
        clock_gettime(CLOCK_MONOTONIC, &t1); tl += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) h = (h ^ o[i]) * 1099511628211ull;
        if (st != VOLCOMP_OK || n != gold[k].n || h != gold[k].h) { fprintf(stderr, "FAIL chunk %d: lossless %zu bytes hash %016llx, expected %zu / %016llx (st %d)\n", k, n, (unsigned long long)h, gold[k].n, (unsigned long long)gold[k].h, (int)st); fails++; }
        if (st == VOLCOMP_OK && (volcomp_decode(o, n, d, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK || memcmp(c, d, VOLCOMP_CHUNK_VOXELS))) { fprintf(stderr, "FAIL chunk %d: lossless round trip\n", k); fails++; }
        {   /* the sink decoder: the whole chunk, then a box cutting blocks */
            uint32_t mode = 0; vll_parsed lp;
            if (st == VOLCOMP_OK && vll_chunk_mode(o, n, &mode) == VOLCOMP_OK && mode == VLL_MODE_CODED && vll_parse(o, n, &lp) == VOLCOMP_OK) {
                vfx_sink K = {d, 0, 128 * 128, 128, {0, 0, 0}, {128, 128, 128}}; memset(d, 0xa5, VOLCOMP_CHUNK_VOXELS); int bad = 0;
                for (uint32_t s = 0; s < VF_NSUB; s++) bad |= vllx_decode_sub_sink(&lp, s, &K) != VOLCOMP_OK;
                if (bad || memcmp(c, d, VOLCOMP_CHUNK_VOXELS)) { fprintf(stderr, "FAIL chunk %d: lossless sink decode\n", k); fails++; }
                static uint8_t box[50 * 61 * 70]; vfx_sink B2 = {box, 0, 61 * 70, 70, {9, 30, 3}, {59, 91, 73}}; bad = 0;
                B2.off = -(ptrdiff_t)(9 * 61 * 70 + 30 * 70 + 3);
                for (uint32_t s = 0; s < VF_NSUB; s++) bad |= vllx_decode_sub_sink(&lp, s, &B2) != VOLCOMP_OK;
                for (int z = 9; z < 59 && !bad; z++) for (int y = 30; y < 91; y++) bad |= memcmp(box + ((size_t)(z - 9) * 61 + (y - 30)) * 70, c + ((size_t)z * 128 + y) * 128 + 3, 70) != 0;
                if (bad) { fprintf(stderr, "FAIL chunk %d: lossless sink decode of a box\n", k); fails++; }
            }
        }
        for (int qi = 0; qi < 3; qi++) {
            const float q = qi == 0 ? 2.f : qi == 1 ? 8.f : 32.f; size_t m = 0;
            if (volcomp_encode(c, q, o, VOLCOMP_ENCODE_BOUND, &m) != VOLCOMP_OK || volcomp_decode(o, m, d, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK) { fprintf(stderr, "FAIL chunk %d: q %g encode / decode\n", k, q); fails++; }
        }
    }
    if (fails) { fprintf(stderr, "test_volcomp_vll: %d failures\n", fails); return 1; }
    printf("test_volcomp_vll: 12 chunks byte-identical to the old lossless encoder (%.0f ms), lossy q 2 / 8 / 32 round trips: OK\n", tl * 1e3);
    return 0;
}
