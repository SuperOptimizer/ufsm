/* lcodec (the learned input codec, stage 1; src/lcodec.h), on the CPU:
   - the loss gradient (rate: Laplace mass of the noisy channels; distortion through the linear synthesis; the analysis MLP) against
     central differences in every parameter block;
   - at init (W2 0, Wl = I / s, S = I s, mean step s) the codec is the KLT at step s: its indices equal latvol's lv_forward on a CT chunk
     (up to rounding ties: <= 0.1 % of the values), and its latvol rows are the KLT rows, so lv_inverse rebuilds the same bytes;
   - a codec with a random nonlinear branch: its indices through lv_encode_chunk / lv_decode_chunk round-trip exactly (the store is
     latvol's format), and lv_inverse of them matches the synthesis S^T q in float to the byte rounding;
   - lcodec_write / lcodec_read: the same parameters and hash. Usage: test_latcodec DIR */
#include "lcodec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static uint64_t S_ = 88172645463325252ull;
static double rnd(void) { S_ ^= S_ << 13; S_ ^= S_ >> 7; S_ ^= S_ << 17; return (double)(S_ >> 11) * 0x1p-53; }
static void basis(lat_basis *lb) {   /* row 0 the mean direction, then Gram-Schmidt of random rows */
    double b[64][64];
    for (int k = 0; k < 64; k++) b[0][k] = 0.125;
    for (int r = 1; r < 64; r++) {
        for (int k = 0; k < 64; k++) b[r][k] = rnd() - 0.5;
        for (int p = 0; p < r; p++) { double d = 0; for (int k = 0; k < 64; k++) d += b[r][k] * b[p][k]; for (int k = 0; k < 64; k++) b[r][k] -= d * b[p][k]; }
        double n = 0; for (int k = 0; k < 64; k++) n += b[r][k] * b[r][k]; n = sqrt(n); for (int k = 0; k < 64; k++) b[r][k] /= n;
    }
    for (int r = 0; r < 64; r++) { lb->sigma[r] = (float)(r ? 30.0 / r : 100); for (int k = 0; k < 64; k++) lb->b[r][k] = (float)b[r][k]; }
}
/* a 128^3 chunk: smooth layers + noise, an air corner */
static void chunk(uint8_t *ct) {
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
        double v = 90 + 50 * sin(0.21 * z + 0.05 * x) + 20 * cos(0.13 * y) + 12 * (rnd() - 0.5);
        if (z < 20 && y < 20 && x < 20) v = 0;
        ct[((size_t)z * 128 + y) * 128 + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}
static void coeffs(const uint8_t *ct, const lat_basis *lb, float sd, float *xg, uint8_t *any) {   /* 32^3 cells */
    for (int c = 0; c < 32768; c++) {
        const int z = c / 1024, y = c / 32 % 32, x = c % 32; double v[64]; int a = 0;
        for (int k = 0; k < 64; k++) { const uint8_t b = ct[((size_t)(4 * z + (k >> 4)) * 128 + 4 * y + (k >> 2 & 3)) * 128 + 4 * x + (k & 3)]; v[k] = b; a |= b; }
        any[c] = a != 0;
        for (int j = 0; j < 64; j++) { double s = 0; for (int k = 0; k < 64; k++) s += (double)lb->b[j][k] * v[k]; xg[(size_t)c * 64 + j] = (float)(s / sd); }
    }
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_latcodec DIR\n"); return 2; }
    lat_basis lb; basis(&lb);
    /* 1. gradient vs central differences */
    { lcodec lc; float sig[64]; for (int j = 0; j < 64; j++) sig[j] = 1.f + (float)j / 16;
      CHECK(!lcodec_init(&lc, 6, 8, 5, 20.f, 0.7f, sig, 0.3f, 3), "init");   /* (tier b on: its blocks too) */
      for (size_t i = 0; i < lc.np; i++) lc.p[i] += (float)(0.2 * (rnd() - 0.5));   /* (every block away from its init: W2 nonzero) */
      enum { N = 3 }; const int K = lc.cl - 1; float f[N * LCD_NF], x[N * 64], u[N * 8];
      for (int i = 0; i < N * LCD_NF; i++) f[i] = (float)(2 * rnd() - 1);
      for (int i = 0; i < N * 64; i++) x[i] = (float)(3 * rnd() - 1.5);
      for (int i = 0; i < N; i++) x[i * 64] = (float)(8 + 4 * rnd());   /* (a material mean: the context's quantised means away from 0) */
      for (int i = 0; i < N * K; i++) u[i] = (float)(rnd() - 0.5);
      float *g = calloc(lc.np, 4); double R, D; const double lam = 0.02;
      lcodec_loss_grad(&lc, N, f, x, u, lam, g, &R, &D);
      const lcodec_off o = lcodec_offs(lc.cl, lc.hid, lc.hid2); const size_t blk[9][2] = {{o.Wl, o.W1}, {o.W1, o.b1}, {o.b1, o.W2}, {o.W2, o.S}, {o.S, o.lb}, {o.lb, o.V1}, {o.V1, o.c1}, {o.c1, o.V2}, {o.V2, o.end}};
      const char *nm[9] = {"Wl", "W1", "b1", "W2", "S", "lb", "V1", "c1", "V2"}; int nbad = 0, ntest = 0;
      for (int b = 0; b < 9; b++) for (int t = 0; t < 6; t++) {
          const size_t i = blk[b][0] + (size_t)(rnd() * (double)(blk[b][1] - blk[b][0])); const float v = lc.p[i], e = 2e-3f; float *gd = calloc(lc.np, 4);
          lc.p[i] = v + e; const double lp = lcodec_loss_grad(&lc, N, f, x, u, lam, gd, &R, &D);
          lc.p[i] = v - e; const double lm = lcodec_loss_grad(&lc, N, f, x, u, lam, gd, &R, &D); lc.p[i] = v; free(gd);
          const double fd = (lp - lm) / (2 * e); ntest++;
          if (fabs(fd - g[i]) > 2e-2 * (fabs(fd) + 1e-3)) { nbad++; fprintf(stderr, "  %s[%zu]: analytic %.6g, central difference %.6g\n", nm[b], i - blk[b][0], g[i], fd); }
      }
      CHECK(nbad == 0, "gradient: %d of %d entries off", nbad, ntest);
      free(g); lcodec_free(&lc); }
    /* 2. at init: the KLT at step s */
    uint8_t *ct = malloc(1 << 21), *any = malloc(32768), *r1 = malloc(1 << 21), *r2 = malloc(1 << 21); float *xg = malloc((size_t)32768 * 64 * 4); chunk(ct);
    const float sd = 33.f, s = 0.5f; const int C = 20; coeffs(ct, &lb, sd, xg, any);
    int16_t *qk = malloc((size_t)32768 * C * 2), *ql = malloc((size_t)32768 * C * 2), *qd = malloc((size_t)32768 * C * 2);
    { lv_codec kl; CHECK(!lv_codec_init(&kl, C, &lb.b[0][0], s, sd, nullptr, 0, nullptr), "klt codec"); lv_forward(&kl, ct, qk);
      lcodec lc; CHECK(!lcodec_init(&lc, C, 16, 0, sd, s, nullptr, 0.f, 5), "init 2"); float f[LCD_NF];
      for (int c = 0; c < 32768; c++) { lcodec_feat(&lc, xg, 32, 32, 32, c / 1024, c / 32 % 32, c % 32, f); lcodec_encode_cell(&lc, f, xg + (size_t)c * 64, any[c], ql + (size_t)c * C); }
      size_t nd = 0, big = 0; for (size_t i = 0; i < (size_t)32768 * C; i++) { nd += qk[i] != ql[i]; big += abs(qk[i] - ql[i]) > 1; }
      CHECK(big == 0 && nd * 1000 <= (size_t)32768 * C, "init = KLT: %zu indices differ (%zu by more than 1) of %d", nd, big, 32768 * C);
      lv_codec lv; CHECK(!lcodec_lv(&lc, &lb, nullptr, &lv), "lcodec_lv");
      int rows = 0; for (int c = 0; c < C; c++) for (int k = 0; k < 64; k++) rows += lv.bi[c][k] != kl.bi[c][k];
      CHECK(rows == 0, "init: %d latvol row entries differ from the KLT's", rows);
      lv_inverse(&kl, qk, r1); lv_inverse(&lv, qk, r2); CHECK(!memcmp(r1, r2, 1 << 21), "init: lv_inverse differs from the KLT's");
      printf("test_latcodec: init = KLT step %.2f: %zu of %d indices differ by a rounding tie\n", s, nd, 32768 * C);
      lv_codec_free(&kl); lv_codec_free(&lv); lcodec_free(&lc); }
    /* 3. a nonlinear codec: the latvol stream round trip, the rebuild = the synthesis */
    { lcodec lc; CHECK(!lcodec_init(&lc, C, 32, 0, sd, s, nullptr, 0.4f, 7), "init 3");
      const lcodec_off o = lcodec_offs(C, 32, 0); for (size_t i = o.W2; i < o.S; i++) lc.p[i] = (float)(0.3 * (rnd() - 0.5));
      for (size_t i = o.S; i < o.lb; i++) lc.p[i] += (float)(0.05 * (rnd() - 0.5));
      float f[LCD_NF];
      for (int c = 0; c < 32768; c++) { lcodec_feat(&lc, xg, 32, 32, 32, c / 1024, c / 32 % 32, c % 32, f); lcodec_encode_cell(&lc, f, xg + (size_t)c * 64, any[c], ql + (size_t)c * C); }
      lv_codec lv; CHECK(!lcodec_lv(&lc, &lb, nullptr, &lv), "lcodec_lv 3");
      uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist); lv_hist(&lv, ql, hist); lv_codec lt; CHECK(!lcodec_lv(&lc, &lb, hist, &lt), "lcodec_lv hist");
      uint8_t *enc = malloc(LV_CHUNK_BOUND); size_t len = 0;
      CHECK(!lv_encode_chunk(&lt, ql, enc, &len) && len > 0, "encode");
      CHECK(!lv_decode_chunk(&lt, enc, len, qd, nullptr) && !memcmp(qd, ql, (size_t)32768 * C * 2), "the latvol stream does not round-trip the learned indices");
      lv_inverse(&lt, qd, r1); size_t off = 0, nm = 0;
      for (int c = 0; c < 32768; c++) { const int16_t *q = ql + (size_t)c * C; if (!q[0]) continue; nm++;
          const int z = c / 1024, y = c / 32 % 32, x = c % 32;
          for (int k = 0; k < 64; k++) { double v = q[0] * lc.mstep * lb.b[0][k];
              for (int cc = 1; cc < C; cc++) for (int j = 0; j < LCD_NAX; j++) v += (double)q[cc] * lc.p[o.S + (size_t)(cc - 1) * LCD_NAX + j] * lb.b[1 + j][k];
              v *= sd; v = v < 1 ? 1 : v > 255 ? 255 : rint(v);
              off += fabs(v - r1[((size_t)(4 * z + (k >> 4)) * 128 + 4 * y + (k >> 2 & 3)) * 128 + 4 * x + (k & 3)]) > 1; } }
      CHECK(nm > 0 && off * 1000 <= nm * 64, "rebuild vs the synthesis: %zu of %zu voxels off by more than 1", off, nm * 64);
      printf("test_latcodec: nonlinear codec: %zu stream bytes for %zu material cells, round trip exact, %zu rebuild voxels off\n", len, nm, off);
      char p[1024]; snprintf(p, sizeof p, "%s/t.lcd", argv[1]); lcodec r;
      CHECK(!lcodec_write(&lc, p) && !lcodec_read(&r, p) && r.np == lc.np && !memcmp(r.p, lc.p, lc.np * 4) && lcodec_hash(&r) == lcodec_hash(&lc), "write / read");
      if (r.p) lcodec_free(&r);
      /* tier b: a file round trip, and with V2 0 its rebuild is the linear one (lcodec_rebuild = lv_inverse to a byte) */
      lcodec lb2; CHECK(!lcodec_init(&lb2, C, 32, 8, sd, lc.mstep, nullptr, 0.4f, 9), "init tier b"); memcpy(lb2.p, lc.p, lc.np * 4);
      snprintf(p, sizeof p, "%s/t2.lcd", argv[1]);
      CHECK(!lcodec_write(&lb2, p) && !lcodec_read(&r, p) && r.hid2 == 8 && r.np == lb2.np && !memcmp(r.p, lb2.p, lb2.np * 4), "tier b write / read");
      if (r.p) lcodec_free(&r);
      { size_t o2 = 0; float cx[LCD_NCX], ff[LCD_NF]; uint8_t v[64];
        for (int c = 0; c < 32768; c++) { const int16_t *q = ql + (size_t)c * C; const int z = c / 1024, y = c / 32 % 32, x = c % 32;
            lcodec_feat(&lb2, xg, 32, 32, 32, z, y, x, ff); lcodec_ctx(&lb2, ff, xg[(size_t)c * 64], cx); lcodec_rebuild(&lb2, &lb, q, cx, v);
            for (int k = 0; k < 64; k++) o2 += abs((int)v[k] - r1[((size_t)(4 * z + (k >> 4)) * 128 + 4 * y + (k >> 2 & 3)) * 128 + 4 * x + (k & 3)]) > 1; }
        CHECK(o2 * 1000 <= (size_t)32768 * 64, "tier b with V2 0: %zu rebuild voxels off the linear one", o2); }
      lcodec_free(&lb2);
      free(enc); free(hist); lv_codec_free(&lv); lv_codec_free(&lt); lcodec_free(&lc); }
    free(ct); free(any); free(r1); free(r2); free(xg); free(qk); free(ql); free(qd);
    if (fails) { fprintf(stderr, "test_latcodec: %d failures\n", fails); return 1; }
    printf("test_latcodec: ok\n");
    return 0;
}
