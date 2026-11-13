/* 2:4 sparse MX MMA on sm_120a: operand / metadata / scale layout of mma.sp m16n8k128 kind::mxf4(nvf4) and m16n8k64 kind::mxf8f6f4
   against an fp64 host reference (random sparse operands, every register of every lane), then the MMA-only throughput dense vs
   sparse. Layout hypothesis (bytes as the sparse elements: an e2m1 byte = a pair of K, so e2m1 sparsity is pair-wise 4:8):
     A compressed 16 x 32 B: reg r of lane (g, t) = row g + 8 (r & 1), compressed bytes 16 (r >> 1) + 4 t + 0..3
     B logical 64 B x 8 (dense: 32 B): reg r = column g, bytes 16 r + 4 t + 0..3
     metadata: one nibble (i0 | i1 << 2, i0 < i1) per (row, 4-byte group), lane / row / group mapping searched among 4 candidates
     scales: A row g lane 4g, row g + 8 lane 4g + 1, B column g lane 4g; byte j = K block j, block width searched (16 / 32 B)
   Discovery tables (UFSM_SP24_DISCOVER=1, or when no candidate matches) print the raw one-hot responses. */
#include "lp_sp24.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t rs = 88172645463325252ull;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 32); }
static const double e2m1v[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
static double dec_e2m1(unsigned n) { return (n & 8 ? -1 : 1) * e2m1v[n & 7]; }
static double dec_e4m3(unsigned b) { int s = b >> 7, e = (b >> 3) & 15, m = b & 7; double v = e ? ldexp(1 + m / 8.0, e - 7) : ldexp(m / 8.0, -6); return s ? -v : v; }
static unsigned rnd_e4m3(void) { return (rnd() & 0x80) | ((5 + rnd() % 5) << 3) | (rnd() & 7); }   /* |v| in [2^-2, 2^2 (1 + 7/8)] */
static unsigned rnd_byte(int f4) { if (!f4) return rnd_e4m3(); unsigned lo = rnd() & 15, hi = rnd() & 15; return lo | hi << 4; }

typedef struct { int v, sparse, f4, KB, nb; const char *name; } var_t;   /* KB: logical bytes of K; nb: B registers */
static const var_t VARS[5] = {
    {0, 0, 1, 32, 2, "dense  m16n8k64  mxf4 2X     (e2m1)"},
    {1, 1, 1, 64, 4, "sparse m16n8k128 mxf4 2X     (e2m1)"},
    {2, 1, 1, 64, 4, "sparse m16n8k128 mxf4nvf4 4X (e2m1)"},
    {3, 0, 0, 32, 2, "dense  m16n8k32  mxf8f6f4 1X (e4m3)"},
    {4, 1, 0, 64, 4, "sparse m16n8k64  mxf8f6f4 1X (e4m3)"},
};
/* metadata lane candidates: lane (g, t) holds row g + 8 * rowbit(t), groups 8 * grpbit(t) + 0..7 */
static int meta_row(int mc, int t) { return mc == 0 ? t >> 1 : mc == 1 ? t & 1 : mc == 2 ? t >> 1 : t & 1; }
static int meta_grp(int mc, int t) { return mc == 0 ? t & 1 : mc == 1 ? t >> 1 : t & 1; }   /* mc 2 / 3: nibble order reversed (below) */

typedef struct { uint8_t A[16][64], B[64][8]; int sa[16][4], sb[8][4]; } prob_t;   /* logical operands (bytes), scale exponents */
static void make_prob(prob_t *p, const var_t *V, uint8_t sel[16][16]) {
    memset(p, 0, sizeof *p);
    for (int m = 0; m < 16; m++) for (int G = 0; G < V->KB / 4; G++) {
        if (V->sparse) {
            static const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
            int c = rnd() % 6; sel[m][G] = (uint8_t)c;
            p->A[m][4 * G + pairs[c][0]] = (uint8_t)rnd_byte(V->f4); p->A[m][4 * G + pairs[c][1]] = (uint8_t)rnd_byte(V->f4);
        } else for (int j = 0; j < 4; j++) p->A[m][4 * G + j] = (uint8_t)rnd_byte(V->f4);
    }
    for (int k = 0; k < V->KB; k++) for (int n = 0; n < 8; n++) p->B[k][n] = (uint8_t)rnd_byte(V->f4);
    for (int m = 0; m < 16; m++) for (int j = 0; j < 4; j++) p->sa[m][j] = 125 + rnd() % 5;
    for (int n = 0; n < 8; n++) for (int j = 0; j < 4; j++) p->sb[n][j] = 125 + rnd() % 5;
}
static void host_ref(const prob_t *p, const var_t *V, int sbb, double C[16][8], double M[16][8]) {
    for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) {
        double s = 0, a = 0;
        for (int kb = 0; kb < V->KB; kb++) {
            double sc = ldexp(1, p->sa[m][kb / sbb] - 127 + p->sb[n][kb / sbb] - 127);
            double t = V->f4 ? dec_e2m1(p->A[m][kb] & 15) * dec_e2m1(p->B[kb][n] & 15) + dec_e2m1(p->A[m][kb] >> 4) * dec_e2m1(p->B[kb][n] >> 4)
                             : dec_e4m3(p->A[m][kb]) * dec_e4m3(p->B[kb][n]);
            s += t * sc; a += fabs(t) * sc;
        }
        C[m][n] = s; M[m][n] = a;
    }
}
static void pack(const prob_t *p, const var_t *V, uint8_t sel[16][16], int mc, unsigned in[32][11]) {
    static const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    memset(in, 0, 32 * 11 * 4);
    for (int l = 0; l < 32; l++) {
        const int g = l >> 2, t = l & 3;
        for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) {
            const int m = g + 8 * (r & 1), cb = 16 * (r >> 1) + 4 * t + j;
            unsigned byte;
            if (V->sparse) { int G = cb >> 1, s = cb & 1; byte = p->A[m][4 * G + pairs[sel[m][G]][s]]; }
            else byte = p->A[m][cb];
            in[l][r] |= byte << (8 * j);
        }
        for (int r = 0; r < V->nb; r++) for (int j = 0; j < 4; j++) in[l][4 + r] |= (unsigned)p->B[16 * r + 4 * t + j][g] << (8 * j);
        if (V->sparse) {
            const int m = g + 8 * meta_row(mc, t), G0 = 8 * meta_grp(mc, t);
            for (int q = 0; q < 8; q++) {
                const int G = G0 + (mc >= 2 ? 7 - q : q), c = sel[m][G];
                in[l][8] |= (unsigned)(pairs[c][0] | pairs[c][1] << 2) << (4 * q);
            }
        }
    }
    for (int g = 0; g < 8; g++) for (int j = 0; j < 4; j++) {
        in[4 * g][9] |= (unsigned)p->sa[g][j] << (8 * j); in[4 * g + 1][9] |= (unsigned)p->sa[g + 8][j] << (8 * j);
        in[4 * g][10] |= (unsigned)p->sb[g][j] << (8 * j);
    }
}
static double run_err(const var_t *V, int mc, int sbb, int trials) {   /* worst |C - ref| / sum |terms| over trials */
    double worst = 0;
    for (int tr = 0; tr < trials; tr++) {
        prob_t p; uint8_t sel[16][16]; unsigned in[32][11]; float out[32][4]; double C[16][8], M[16][8];
        make_prob(&p, V, sel); pack(&p, V, sel, mc, in);
        if (lp_sp24_mma_probe(V->v, &in[0][0], &out[0][0])) return 1e9;
        host_ref(&p, V, sbb, C, M);
        for (int l = 0; l < 32; l++) for (int i = 0; i < 4; i++) {
            const int m = (l >> 2) + 8 * (i >> 1), n = 2 * (l & 3) + (i & 1);
            const double e = fabs(out[l][i] - C[m][n]) / (M[m][n] > 0 ? M[m][n] : 1);
            if (e > worst) worst = e;
        }
    }
    return worst;
}
/* raw responses: B one-hot (every byte 1.0 / 1.0 pair) against all-ones compressed A under one global metadata nibble */
static void discover(const var_t *V) {
    const unsigned one = V->f4 ? 0x22 : 0x38, ones = one * 0x01010101u;
    printf("  discovery %s: B byte (lane.reg.byte) -> column n, rows lit, under metadata nibble 4 / 8 / E\n", V->name);
    const unsigned metas[3] = {0x44444444u, 0x88888888u, 0xEEEEEEEEu};
    for (int l = 0; l < 32; l++) for (int r = 0; r < V->nb; r++) for (int j = 0; j < 4; j++) {
        printf("   B %2d.%d.%d:", l, r, j);
        for (int mi = 0; mi < (V->sparse ? 3 : 1); mi++) {
            unsigned in[32][11] = {{0}}; float out[32][4];
            for (int x = 0; x < 32; x++) { for (int q = 0; q < 4; q++) in[x][q] = ones; in[x][8] = metas[mi]; in[x][9] = in[x][10] = 0x7f7f7f7fu; }
            in[l][4 + r] = one << (8 * j);
            lp_sp24_mma_probe(V->v, &in[0][0], &out[0][0]);
            int n = -1, rows = 0;
            for (int x = 0; x < 32; x++) for (int i = 0; i < 4; i++) if (out[x][i] != 0) { rows++; n = 2 * (x & 3) + (i & 1); }
            printf("  n%d rows %2d", n, rows);
        }
        printf("\n");
    }
    if (!V->sparse) return;
    printf("  discovery: metadata (lane, nibble) 4 -> E: the B bytes (lane.reg.byte) and output rows that change\n");
    for (int l = 0; l < 4; l++) for (int q = 0; q < 8; q++) {   /* lanes of group 0 (rows 0 / 8) */
        printf("   E %2d.%d:", l, q);
        for (int bl = 0; bl < 32; bl++) for (int r = 0; r < V->nb; r++) for (int j = 0; j < 4; j++) {
            unsigned in[32][11] = {{0}}; float out[32][4];
            for (int x = 0; x < 32; x++) { for (int a = 0; a < 4; a++) in[x][a] = ones; in[x][8] = 0x44444444u; in[x][9] = in[x][10] = 0x7f7f7f7fu; }
            in[l][8] = (0x44444444u & ~(15u << (4 * q))) | (0xEu << (4 * q));
            in[bl][4 + r] = one << (8 * j);
            lp_sp24_mma_probe(V->v, &in[0][0], &out[0][0]);
            int rowmask = 0;
            for (int x = 0; x < 32; x++) for (int i = 0; i < 4; i++) if (out[x][i] != 0) rowmask |= 1 << ((x >> 2) + 8 * (i >> 1));
            if (rowmask) printf(" %d.%d.%d:%04x", bl, r, j, rowmask);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int bad = 0;
    const int disc = getenv("UFSM_SP24_DISCOVER") ? atoi(getenv("UFSM_SP24_DISCOVER")) : 0;
    int found[5][2] = {{0}};
    for (int vi = 0; vi < 5; vi++) {
        const var_t *V = &VARS[vi];
        double best = 1e30; int bmc = -1, bsb = -1;
        for (int mc = 0; mc < (V->sparse ? 4 : 1); mc++) for (int sbb = 16; sbb <= (V->f4 || !V->sparse ? 32 : 64); sbb *= 2) {
            if (!V->f4 && !V->sparse && sbb != 32) continue;
            const double e = run_err(V, mc, sbb, 20);
            if (e < best) { best = e; bmc = mc; bsb = sbb; }
        }
        const int ok = best < 1e-6;
        printf("%s: %s (metadata candidate %d, scale block %d B, worst err %.3g)\n", V->name, ok ? "layout ok" : "NO MATCH", bmc, bsb, best);
        found[vi][0] = bmc; found[vi][1] = bsb;
        if (!ok) bad++;
        if (!ok || disc) discover(V);
    }
    /* throughput: random valid operands of the matched layout; 4 blocks of 8 warps per SM-ish (sweep) */
    if (!bad || disc) {
        double t[5];
        for (int vi = 0; vi < 5; vi++) {
            const var_t *V = &VARS[vi];
            prob_t p; uint8_t sel[16][16]; unsigned in[32][11];
            make_prob(&p, V, sel); pack(&p, V, sel, found[vi][0] < 0 ? 0 : found[vi][0], in);
            double b = 0;
            for (int blocks = 36; blocks <= 288; blocks *= 2) { double x = lp_sp24_mma_bench(V->v, &in[0][0], blocks, 4096); if (x > b) b = x; }
            t[vi] = b;
            printf("bench %s: %.1f dense-equivalent TFLOP/s\n", V->name, b);
        }
        printf("sparse / dense: e2m1 2X %.2fx, e2m1 nvf4 4X %.2fx, e4m3 %.2fx\n", t[1] / t[0], t[2] / t[0], t[4] / t[3]);
    }
    printf("%s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
