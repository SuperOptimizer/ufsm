#include "fuse.h"
#include "band.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void fuse_phase(const uint8_t *const *in, int n, size_t len, float tol_turns, uint8_t *out, size_t *stats, size_t *hist) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static int init;
    if (!init) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = cosf(2.f * (float)M_PI * k / PHASE_PERIOD); sn[k] = sinf(2.f * (float)M_PI * k / PHASE_PERIOD); } init = 1; }
    const float tol = tol_turns * PHASE_PERIOD;   /* in codes */
    size_t st[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < len; i++) {
        int m = 0, c[FUSE_MAX]; float sx = 0, sy = 0;
        for (int k = 0; k < n && m < FUSE_MAX; k++) { const uint8_t v = in[k][i]; if (v < PHASE_PERIOD) { c[m++] = v; sx += cs[v]; sy += sn[v]; } }
        if (!m) { out[i] = PHASE_UNKNOWN; st[3]++; continue; }
        if (m == 1) { out[i] = (uint8_t)c[0]; st[0]++; continue; }
        int dmax = 0;
        for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) {
            int d = abs(c[a] - c[b]); if (d > PHASE_PERIOD / 2) d = PHASE_PERIOD - d;
            if (d > dmax) dmax = d;
        }
        if (hist) hist[dmax]++;
        if ((float)dmax > tol) { out[i] = PHASE_UNKNOWN; st[2]++; continue; }
        float a = atan2f(sy, sx); if (a < 0) a += 2.f * (float)M_PI;
        int q = (int)lroundf(a * PHASE_PERIOD / (2.f * (float)M_PI)); out[i] = (uint8_t)(q % PHASE_PERIOD); st[1]++;
    }
    if (stats) for (int k = 0; k < 4; k++) stats[k] += st[k];
}

void fuse_recto(const uint8_t *const *in, int n, const uint8_t *ct, int64_t d0, int64_t d1, int64_t d2, int block, int dup,
                uint8_t *out, size_t *stats) {
    /* score[k][b]: the mean CT over set k's surface voxels in block b (its trace's fit to the papyrus ridge there); a surface voxel of
       set k is dropped when another set has a surface voxel within dup whose block score is higher (ties: the lower set index).
       Sparse: per block the sets present (bits); only blocks with a surface are visited, and a voxel whose neighbour blocks hold no
       other set is unique coverage without the dup^3 scan (the scan runs only where traces of two sets come close) */
    const size_t len = (size_t)d0 * d1 * d2; const int nn = n < FUSE_MAX ? n : FUSE_MAX;
    const int64_t b0 = (d0 + block - 1) / block, b1 = (d1 + block - 1) / block, b2 = (d2 + block - 1) / block; const size_t nb = (size_t)b0 * b1 * b2;
    memset(out, 0, len);
    size_t st[3] = {0, 0, 0};
    if (nn == 1) {   /* one set: its surface as it is (ct may be null) */
        for (size_t i = 0; i < len; i++) if (in[0][i]) { out[i] = 255; st[1]++; }
        if (stats) stats[1] += st[1];
        return;
    }
    double *sum = calloc(nb * nn, sizeof *sum); size_t *cnt = calloc(nb * nn, sizeof *cnt); float *score = malloc(nb * nn * sizeof *score);
    uint64_t *bs = calloc(nb, sizeof *bs), *bd = calloc(nb, sizeof *bd);
    if (!sum || !cnt || !score || !bs || !bd) { free(sum); free(cnt); free(score); free(bs); free(bd); return; }
    for (int64_t z = 0; z < d0; z++) for (int64_t y = 0; y < d1; y++) {
        const size_t r0 = ((size_t)z * d1 + y) * d2, rb = ((size_t)(z / block) * b1 + y / block) * b2;
        for (int k = 0; k < nn; k++) { const uint8_t *row = in[k] + r0;
            for (int64_t x = 0; x < d2; x++) if (row[x]) { const size_t b = rb + x / block; sum[b * nn + k] += ct[r0 + x]; cnt[b * nn + k]++; bs[b] |= 1ull << k; } }
    }
    for (size_t b = 0; b < nb * nn; b++) score[b] = cnt[b] ? (float)(sum[b] / (double)cnt[b]) : -1.f;
    const int64_t R = (dup + block - 1) / block;   /* neighbour blocks a dup cube can reach */
    for (int64_t bz = 0; bz < b0; bz++) for (int64_t by = 0; by < b1; by++) for (int64_t bx = 0; bx < b2; bx++) {
        uint64_t v = 0;
        for (int64_t z = bz - R; z <= bz + R; z++) for (int64_t y = by - R; y <= by + R; y++) for (int64_t x = bx - R; x <= bx + R; x++)
            if (z >= 0 && z < b0 && y >= 0 && y < b1 && x >= 0 && x < b2) v |= bs[((size_t)z * b1 + y) * b2 + x];
        bd[((size_t)bz * b1 + by) * b2 + bx] = v;
    }
    for (int64_t bz = 0; bz < b0; bz++) for (int64_t by = 0; by < b1; by++) for (int64_t bx = 0; bx < b2; bx++) {
        const size_t b = ((size_t)bz * b1 + by) * b2 + bx; if (!bs[b]) continue;
        const int64_t z1 = (bz + 1) * block < d0 ? (bz + 1) * block : d0, y1 = (by + 1) * block < d1 ? (by + 1) * block : d1, x1 = (bx + 1) * block < d2 ? (bx + 1) * block : d2;
        for (int64_t z = bz * block; z < z1; z++) for (int64_t y = by * block; y < y1; y++) for (int64_t x = bx * block; x < x1; x++) {
            const size_t i = ((size_t)z * d1 + y) * d2 + x;
            for (int k = 0; k < nn; k++) {
                if (!(bs[b] >> k & 1) || !in[k][i]) continue;
                if (!(bd[b] & ~(1ull << k))) { if (!out[i]) { out[i] = 255; st[1]++; } continue; }   /* no other set near */
                const float sk = score[b * nn + k]; int beaten = 0, had_other = 0;
                for (int64_t zz = z - dup; zz <= z + dup && !beaten; zz++) { if (zz < 0 || zz >= d0) continue;
                    for (int64_t yy = y - dup; yy <= y + dup && !beaten; yy++) { if (yy < 0 || yy >= d1) continue;
                        for (int64_t xx = x - dup; xx <= x + dup && !beaten; xx++) { if (xx < 0 || xx >= d2) continue;
                            const size_t j = ((size_t)zz * d1 + yy) * d2 + xx, bj = ((size_t)(zz / block) * b1 + yy / block) * b2 + xx / block;
                            for (int o = 0; o < nn; o++) { if (o == k || !in[o][j]) continue; had_other = 1;
                                const float so = score[bj * nn + o]; if (so > sk || (so == sk && o < k)) { beaten = 1; break; } } } } }
                if (beaten) { st[2]++; continue; }
                if (!out[i]) { out[i] = 255; st[had_other ? 0 : 1]++; }
            }
        }
    }
    free(sum); free(cnt); free(score); free(bs); free(bd);
    if (stats) for (int k = 0; k < 3; k++) stats[k] += st[k];
}
