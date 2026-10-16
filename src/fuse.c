#include "fuse.h"
#include "band.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void fuse_phase(const uint8_t *const *in, int n, size_t len, float tol_turns, uint8_t *out, size_t *stats) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static int init;
    if (!init) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = cosf(2.f * (float)M_PI * k / PHASE_PERIOD); sn[k] = sinf(2.f * (float)M_PI * k / PHASE_PERIOD); } init = 1; }
    const float tol = tol_turns * PHASE_PERIOD;   /* in codes */
    size_t st[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < len; i++) {
        int m = 0, c[16]; float sx = 0, sy = 0;
        for (int k = 0; k < n && m < 16; k++) { const uint8_t v = in[k][i]; if (v < PHASE_PERIOD) { c[m++] = v; sx += cs[v]; sy += sn[v]; } }
        if (!m) { out[i] = PHASE_UNKNOWN; st[3]++; continue; }
        if (m == 1) { out[i] = (uint8_t)c[0]; st[0]++; continue; }
        int bad = 0;
        for (int a = 0; a < m && !bad; a++) for (int b = a + 1; b < m; b++) {
            int d = abs(c[a] - c[b]); if (d > PHASE_PERIOD / 2) d = PHASE_PERIOD - d;
            if ((float)d > tol) { bad = 1; break; }
        }
        if (bad) { out[i] = PHASE_UNKNOWN; st[2]++; continue; }
        float a = atan2f(sy, sx); if (a < 0) a += 2.f * (float)M_PI;
        int q = (int)lroundf(a * PHASE_PERIOD / (2.f * (float)M_PI)); out[i] = (uint8_t)(q % PHASE_PERIOD); st[1]++;
    }
    if (stats) for (int k = 0; k < 4; k++) stats[k] += st[k];
}

void fuse_recto(const uint8_t *const *in, int n, const uint8_t *ct, int64_t d0, int64_t d1, int64_t d2, int block, int dup,
                uint8_t *out, size_t *stats) {
    /* score[k][b]: the mean CT over set k's surface voxels in block b (its trace's fit to the papyrus ridge there); a surface voxel of
       set k is dropped when another set has a surface voxel within dup whose block score is higher (ties: the lower set index) */
    const size_t len = (size_t)d0 * d1 * d2; const int nn = n < 16 ? n : 16;
    const int64_t b0 = (d0 + block - 1) / block, b1 = (d1 + block - 1) / block, b2 = (d2 + block - 1) / block; const size_t nb = (size_t)b0 * b1 * b2;
    double *sum = calloc(nb * nn, sizeof *sum); size_t *cnt = calloc(nb * nn, sizeof *cnt); float *score = malloc(nb * nn * sizeof *score);
    memset(out, 0, len);
    if (!sum || !cnt || !score) { free(sum); free(cnt); free(score); return; }
    for (int64_t z = 0; z < d0; z++) for (int64_t y = 0; y < d1; y++) for (int64_t x = 0; x < d2; x++) {
        const size_t i = ((size_t)z * d1 + y) * d2 + x, b = ((size_t)(z / block) * b1 + y / block) * b2 + x / block;
        for (int k = 0; k < nn; k++) if (in[k][i]) { sum[b * nn + k] += ct[i]; cnt[b * nn + k]++; }
    }
    for (size_t b = 0; b < nb * nn; b++) score[b] = cnt[b] ? (float)(sum[b] / (double)cnt[b]) : -1.f;
    size_t st[3] = {0, 0, 0};
    for (int64_t z = 0; z < d0; z++) for (int64_t y = 0; y < d1; y++) for (int64_t x = 0; x < d2; x++) {
        const size_t i = ((size_t)z * d1 + y) * d2 + x, b = ((size_t)(z / block) * b1 + y / block) * b2 + x / block;
        for (int k = 0; k < nn; k++) {
            if (!in[k][i]) continue;
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
    free(sum); free(cnt); free(score);
    if (stats) for (int k = 0; k < 3; k++) stats[k] += st[k];
}
