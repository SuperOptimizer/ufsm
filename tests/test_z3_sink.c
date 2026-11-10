/* z3_read's window sink (zarr3.c make_sink, volcomp_fast.h vfx_decode_subs_sink / vllx_decode_sub_sink): lossy, lossless-coded and
   constant chunks decoded straight into the request give the same bytes as decoding whole chunks and copying (UFSM_Z3_SINK=0), for
   boxes that cut blocks, chunks and the array's edge (a shape that is not a multiple of the chunk), with and without the substream
   selection (UFSM_Z3_AUTOSUB). */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char dir[] = "/tmp/test_z3sink_XXXXXX"; if (!mkdtemp(dir)) return 1;
    enum { NZ = 300, NY = 200, NX = 260 };
    const size_t nv = (size_t)NZ * NY * NX;
    uint8_t *d = malloc(nv);
    for (size_t i = 0; i < nv; i++) {
        const int z = (int)(i / ((size_t)NY * NX)), y = (int)(i / NX % NY), x = (int)(i % NX);
        if (z >= 128 && z < 256 && y < 128 && x < 128) { d[i] = 77; continue; }   /* a constant chunk */
        if (x >= 128 && x < 160 && y >= 16 && y < 48) { d[i] = 200; continue; }   /* constant blocks inside coded chunks */
        d[i] = (uint8_t)(100 + 60 * sin(0.13 * x + 0.05 * y) * cos(0.07 * z) + (int)((i * 2654435761u) >> 28));
    }
    int fails = 0, nbox = 0;
    const float qs[2] = {8.f, 0.f};   /* lossy, lossless */
    for (int qi = 0; qi < 2; qi++) {
        char p[512]; snprintf(p, sizeof p, "%s/v%d/0", dir, qi);
        const int64_t sh[3] = {NZ, NY, NX};
        z3w *w = z3w_create(p, sh, 256, qs[qi], 0, nullptr);
        if (!w) { fprintf(stderr, "create failed\n"); return 1; }
        for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 1; sy++) for (int sx = 0; sx < 2; sx++) {   /* 256^3 shards, zero padded */
            uint8_t *sb = calloc((size_t)256 * 256 * 256, 1);
            for (int z = 0; z < 256 && sz * 256 + z < NZ; z++) for (int y = 0; y < 256 && sy * 256 + y < NY; y++) for (int x = 0; x < 256 && sx * 256 + x < NX; x++)
                sb[((size_t)z * 256 + y) * 256 + x] = d[((size_t)(sz * 256 + z) * NY + (sy * 256 + y)) * NX + (sx * 256 + x)];
            if (z3w_write_shard(w, sz, sy, sx, sb, 2)) { fprintf(stderr, "write failed\n"); return 1; }
            free(sb);
        }
        if (z3w_close(w)) { fprintf(stderr, "close failed\n"); return 1; }
        snprintf(p, sizeof p, "%s/v%d", dir, qi);
        store *s = store_open(p); z3 *z = s ? z3_open(s, "0", nullptr) : nullptr;
        if (!z) { fprintf(stderr, "open failed: %s\n", z3_error()); return 1; }
        int64_t boxes[40][6] = {{0, 0, 0, NZ, NY, NX}, {96, 100, 3, 64, 40, 200}, {127, 31, 64, 2, 2, 2}, {16, 32, 0, 16, 32, 128}, {200, 150, 200, 100, 50, 60},
                                {128, 0, 0, 128, 128, 128}, {120, 10, 120, 20, 50, 50}, {250, 180, 240, 60, 30, 40}};
        unsigned long long rs = 0x243f6a8885a308d3ull;
        for (int b = 8; b < 40; b++) for (int dd = 0; dd < 3; dd++) {
            const int L = dd == 0 ? NZ : dd == 1 ? NY : NX;
            rs = rs * 6364136223846793005ull + 1442695040888963407ull; const int n = 1 + (int)((rs >> 33) % 150);
            rs = rs * 6364136223846793005ull + 1442695040888963407ull; boxes[b][dd] = (int64_t)((rs >> 33) % (unsigned)(L - 1)); boxes[b][3 + dd] = n;
            if (boxes[b][dd] + n > L) boxes[b][3 + dd] = L - boxes[b][dd];
        }
        for (int b = 0; b < 40; b++) for (int as = 0; as < 2; as++) {
            const int64_t *o = boxes[b], *n = boxes[b] + 3; const size_t bn = (size_t)n[0] * n[1] * n[2];
            uint8_t *a = malloc(bn), *f = malloc(bn); memset(a, 0xa5, bn); memset(f, 0x5a, bn);
            if (as) setenv("UFSM_Z3_AUTOSUB", "0", 1); else unsetenv("UFSM_Z3_AUTOSUB");
            setenv("UFSM_Z3_SINK", "0", 1); const int ra = z3_read(z, o, n, f, 2);
            unsetenv("UFSM_Z3_SINK"); const int rb = z3_read(z, o, n, a, 2);
            int bad = ra || rb || memcmp(a, f, bn);
            if (!bad && qi == 1 && o[0] + n[0] <= NZ && o[1] + n[1] <= NY && o[2] + n[2] <= NX)   /* lossless, inside the array: the source bytes */
                for (int64_t zz = 0; zz < n[0] && !bad; zz++) for (int64_t yy = 0; yy < n[1] && !bad; yy++)
                    bad = memcmp(a + ((size_t)zz * n[1] + yy) * n[2], d + ((size_t)(o[0] + zz) * NY + (o[1] + yy)) * NX + o[2], (size_t)n[2]) != 0;
            if (bad) { fprintf(stderr, "FAIL q %g box %d (%lld %lld %lld + %lld %lld %lld) autosub %d: sink read differs\n", qs[qi], b, (long long)o[0], (long long)o[1], (long long)o[2], (long long)n[0], (long long)n[1], (long long)n[2], !as); fails++; }
            nbox++; free(a); free(f);
        }
        z3_close(z); store_close(s);
    }
    unsetenv("UFSM_Z3_AUTOSUB");
    free(d);
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_z3_sink: %d of %d FAILED (kept %s)\n", fails, nbox, dir); return 1; }
    if (z3_sink_chunks() == 0) { fprintf(stderr, "test_z3_sink: no chunk went through the sink\n"); return 1; }
    printf("test_z3_sink: ok (%d reads, %llu chunks through the sink)\n", nbox, (unsigned long long)z3_sink_chunks());
    return 0;
}
