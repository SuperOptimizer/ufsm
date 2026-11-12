/* vcgpu (src/vcgpu.h): the GPU's levels (nn_vcg_shard) + vcg_encode_levels give volcomp_encode's stream byte for byte, every chunk of a
   256^3 shard (8 chunks: smooth fields with noise like a head's probabilities, sharp edges, constant blocks, a constant chunk, an all-zero
   chunk, a full-range random chunk), q 1, 2, 8, 30, 255; and a capacity below the total is flagged (the caller encodes from the bytes).
   Usage: test_vcgpu */
#include "nn.h"
#include "vcgpu.h"
#include "../third_party/volcomp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { N = 256 };
static uint64_t rs = 7;
static double rnd(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (double)(rs >> 11) * 0x1p-53; }
int main(void) {
    nn_init(0);
    const size_t n3 = (size_t)N * N * N, cap_e = n3 / 2;   /* (q 1 on the random chunk: ~4000 levels a block) */
    uint8_t *v = malloc(n3), *ch = malloc((size_t)1 << 21), *e0 = malloc(VOLCOMP_ENCODE_BOUND), *e1 = malloc(VOLCOMP_ENCODE_BOUND);
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const int c = (z / 128) * 4 + (y / 128) * 2 + x / 128; double f;
        switch (c) {
        case 0: f = 128 + 120 * sin(x * 0.07) * cos(y * 0.05 + z * 0.03) + 6 * (rnd() - 0.5); break;   /* smooth + noise */
        case 1: f = ((x / 9 + y / 13 + z / 7) & 1) ? 230 : 15; break;                                  /* sharp edges */
        case 2: f = (x & 16) ? 77 : 200 + 50 * rnd(); break;                                            /* constant blocks beside noise */
        case 3: f = 93; break;                                                                          /* a constant chunk */
        case 4: f = 0; break;                                                                           /* all zero */
        case 5: f = 256 * rnd(); break;                                                                 /* full-range random */
        case 6: f = 1 / (1 + exp(-(hypot(x - 192.0, y - 64.0) - 40) * 0.3)) * 255; break;               /* a sigmoid shell */
        default: f = (z % 128) < 64 ? 0 : 255 * (rnd() < 0.02); break;                                  /* half empty, sparse spikes */
        }
        v[((size_t)z * N + y) * N + x] = (uint8_t)(f < 0 ? 0 : f > 255 ? 255 : f);
    }
    vcg_layout L; vcg_layout_of(&L, N, cap_e);
    uint8_t *dv = nn_malloc(n3), *dout = nn_malloc(L.bytes), *hout = malloc(L.bytes); float *drs = nn_malloc(4096 * 4); uint16_t *dsc = nn_malloc(4096 * 2);
    nn_h2d(dv, v, n3);
    float rstep[4096]; uint16_t scan[4096]; int fails = 0;
    static const float Q[] = {1, 2, 8, 30, 255};
    for (int qi = 0; qi < 5; qi++) {
        const float q = Q[qi]; const int fma = vcg_tables(q, rstep, scan);
        nn_h2d(drs, rstep, sizeof rstep); nn_h2d(dsc, scan, sizeof scan);
        if (nn_vcg_shard(dv, N, drs, dsc, fma, dout, (int32_t *)(dout + L.o_lv), (uint16_t *)(dout + L.o_pos), cap_e)) { printf("FAIL nn_vcg_shard\n"); return 1; }
        nn_sync(); nn_d2h(hout, dout, L.bytes);
        uint64_t tot; memcpy(&tot, hout, 8);
        if (tot > cap_e) { printf("FAIL q %g: %llu levels above the capacity %zu\n", q, (unsigned long long)tot, cap_e); return 1; }
        const int32_t *dc = (const int32_t *)(hout + L.o_dc), *lv = (const int32_t *)(hout + L.o_lv); const uint32_t *cnt = (const uint32_t *)(hout + L.o_cnt), *off = (const uint32_t *)(hout + L.o_off);
        const int16_t *cv = (const int16_t *)(hout + L.o_cv); const uint16_t *pos = (const uint16_t *)(hout + L.o_pos);
        int nbad = 0, ncv = 0; size_t bytes = 0;
        for (int c = 0; c < 8; c++) {
            const int cz = c / 4, cy = c / 2 % 2, cx = c % 2;
            for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(ch + ((size_t)z * 128 + y) * 128, v + ((size_t)(cz * 128 + z) * N + cy * 128 + y) * N + cx * 128, 128);
            size_t n0, n1;
            if (volcomp_encode(ch, q, e0, VOLCOMP_ENCODE_BOUND, &n0) != VOLCOMP_OK) { printf("FAIL cpu encode\n"); return 1; }
            const int st = vcg_encode_levels(q, dc + c * 512, cnt + c * 512, off + c * 512, lv, pos, e1, VOLCOMP_ENCODE_BOUND, &n1);
            if (st || n0 != n1 || memcmp(e0, e1, n0)) { nbad++; if (nbad < 3) printf("  q %g chunk %d: status %d, %zu vs %zu bytes\n", q, c, st, n0, n1); }
            for (int b = 0; b < 512; b++) ncv += cv[c * 512 + b] >= 0;
            bytes += n0;
        }
        printf("%s q %-3g: %d of 8 chunks differ from volcomp_encode (%llu AC levels, %d constant blocks, %.1f KB, fused AC quantiser %d)\n", nbad ? "FAIL" : "ok  ", q, nbad, (unsigned long long)tot, ncv, bytes / 1024., fma);
        fails += nbad != 0;
        if (qi == 2) {   /* the levels written straight into pinned host memory: the same */
            uint8_t *hp = nn_host_alloc((size_t)cap_e * 6), *hd = malloc(L.bytes);
            nn_vcg_shard(dv, N, drs, dsc, fma, dout, (int32_t *)hp, (uint16_t *)(hp + cap_e * 4), cap_e); nn_sync();
            const int bad = memcmp(hp, hout + L.o_lv, tot * 4) || memcmp(hp + cap_e * 4, hout + L.o_pos, tot * 2);
            printf("%s q 8, levels into pinned host memory: %s\n", bad ? "FAIL" : "ok  ", bad ? "differ" : "the same"); fails += bad; nn_host_free(hp); free(hd);
        }
        if (qi == 0) {   /* a capacity below the total: flagged */
            vcg_layout L2; vcg_layout_of(&L2, N, 1000); uint8_t *d2 = nn_malloc(L2.bytes);
            nn_vcg_shard(dv, N, drs, dsc, fma, d2, (int32_t *)(d2 + L2.o_lv), (uint16_t *)(d2 + L2.o_pos), 1000); nn_sync(); uint64_t t2; nn_d2h(&t2, d2, 8); nn_free(d2);
            const int bad = t2 != tot; printf("%s capacity 1000: total %llu reported (> capacity)\n", bad ? "FAIL" : "ok  ", (unsigned long long)t2); fails += bad;
        }
    }
    printf("%s\n", fails ? "FAIL" : "test_vcgpu: ok");
    return fails != 0;
}
