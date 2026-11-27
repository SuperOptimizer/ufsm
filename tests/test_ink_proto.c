/* ink-proto (src/inkproto.h, src/inkproto.c, src/nn_proto.cu): the label chunk against a hand-computed case and an independent loop,
   the prototype file round trip and vector, and the two kernels against host references (fp16 features k / 64, exact in fp16; fp16 and
   fp32 logits): the cosine output within 2e-3 in p, the seed sums within 1e-5 relative and the count exact, the window interior honoured
   on a z-offset slab. */
#include "inkproto.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static uint16_t f2h(float v) {   /* exact for 0 and k / 64, |k| <= 1024 (and other values with <= 11 significant bits, normal range) */
    if (v == 0) return 0;
    const uint16_t s = v < 0 ? 0x8000u : 0; int e; const double m = frexp(fabs((double)v), &e);
    return (uint16_t)(s | (uint16_t)((e - 1 + 15) << 10) | (uint16_t)lrint((m * 2 - 1) * 1024));
}
static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; double v;
    if (!e) v = ldexp(m, -24); else v = ldexp(1 + m / 1024.0, e - 15);
    return (float)(s ? -v : v);
}
static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void test_chunk(void) {
    /* hand case: 6 voxels; c over the unmasked ones 100..200 */
    const uint8_t t[6] = {255, 255, 128, 0, 255, 255}, c[6] = {200, 100, 200, 250, 150, 30}, ct[6] = {90, 90, 90, 90, 90, 20};
    uint8_t o[6];
    long k = ink_proto_label_chunk(t, c, ct, 6, 50, 0.5, 0, o);
    /* s: 1, 0, 128/255, masked (t 0), 0.5, masked (CT 20 < 50) */
    CHECK(o[0] == 255 && o[1] == 1 && o[2] == (uint8_t)(1 + lrint(254 * 128 / 255.0)) && o[3] == 0 && o[4] == 128 && o[5] == 0, "score %d %d %d %d %d %d", o[0], o[1], o[2], o[3], o[4], o[5]);
    CHECK(k == 2, "ink count %ld (s 1 and 128/255 > 0.5; 0.5 is not)", k);
    k = ink_proto_label_chunk(t, c, ct, 6, 50, 0.5, 1, o);
    CHECK(o[0] == 254 && o[1] == 0 && o[2] == 254 && o[3] == 255 && o[4] == 0 && o[5] == 255 && k == 2, "prob254 %d %d %d %d %d %d", o[0], o[1], o[2], o[3], o[4], o[5]);
    const uint8_t cf[6] = {120, 120, 120, 120, 120, 120};   /* no contrast: 0 */
    k = ink_proto_label_chunk(t, cf, ct, 6, 50, 0.5, 0, o);
    CHECK(k == 0 && o[0] == 1 && o[4] == 1 && o[3] == 0, "flat chunk");
    /* random chunks against an independent loop */
    enum { n = 4096 }; static uint8_t T[n], C[n], K[n], O[n];
    for (int r = 0; r < 20; r++) {
        for (int i = 0; i < n; i++) { T[i] = rnd() % 4 ? (uint8_t)(rnd() % 256) : 0; C[i] = rnd() % 5 ? (uint8_t)(1 + rnd() % 255) : 0; K[i] = (uint8_t)(rnd() % 256); }
        const int air = (int)(rnd() % 100); const double thr = (rnd() % 90) / 100.0;
        double mn = 1e9, mx = -1e9;
        for (int i = 0; i < n; i++) if (T[i] > 0 && C[i] > 0 && K[i] >= air) { mn = fmin(mn, C[i]); mx = fmax(mx, C[i]); }
        k = ink_proto_label_chunk(T, C, K, n, air, thr, 0, O); long kr = 0; int bad = 0;
        for (int i = 0; i < n; i++) {
            const int m = !(T[i] > 0 && C[i] > 0 && K[i] >= air); double s = m || !(mx > mn) ? 0 : (T[i] / 255.0) * ((C[i] - mn) / (mx - mn));
            if (!m && s > thr) kr++;
            const int e = m ? 0 : (int)(1 + lrint(254 * s)); if (O[i] != e) bad++;
        }
        CHECK(!bad && k == kr, "random chunk %d: %d voxels differ, ink %ld vs %ld", r, bad, k, kr);
    }
}
static void test_file(void) {
    ink_proto_file P = {.C = 3, .n = 10}, Q; float v[3];
    P.raw[0] = 3; P.raw[1] = 4; P.raw[2] = 0; P.unit[0] = 0; P.unit[1] = -2; P.unit[2] = 0;
    const char *tmp = getenv("TMPDIR"); char p[512]; snprintf(p, sizeof p, "%s/test_ink_proto.%d.proto", tmp && *tmp ? tmp : "/tmp", (int)rng);
    CHECK(!ink_proto_write(p, &P) && !ink_proto_read(p, &Q), "file round trip");
    CHECK(Q.C == 3 && Q.n == 10 && Q.raw[1] == 4 && Q.unit[1] == -2, "file contents");
    CHECK(!ink_proto_vector(&Q, 1, v) && fabsf(v[0] - 0.6f) < 1e-6f && fabsf(v[1] - 0.8f) < 1e-6f, "raw vector %g %g", v[0], v[1]);
    CHECK(!ink_proto_vector(&Q, 0, v) && v[1] == -1.f && v[0] == 0.f, "unit vector");
    Q.n = 0; CHECK(ink_proto_vector(&Q, 0, v) == -1, "empty seed refused");
    remove(p);
}
static void test_gpu(void) {
    enum { C = 32, D = 12, H = 10, W = 14, z0 = 3, nz = 6, hz = 4, hy = 2, hx = 3 };
    const size_t nv = (size_t)nz * H * W;
    uint16_t *ft = malloc(nv * C * 2); float *lgf = malloc(nv * 4); uint16_t *lgh = malloc(nv * 2); float pr[C]; double q = 0;
    for (size_t i = 0; i < nv * C; i++) ft[i] = f2h(((int)(rnd() % 257) - 128) / 64.f);
    for (int c = 0; c < C; c++) { pr[c] = ((int)(rnd() % 201) - 100) / 100.f; q += (double)pr[c] * pr[c]; }
    for (int c = 0; c < C; c++) pr[c] = (float)(pr[c] / sqrt(q));
    for (size_t v = 0; v < nv; v++) { const float l = ((int)(rnd() % 129) - 64) / 16.f; lgf[v] = l; lgh[v] = f2h(l); }
    for (size_t v = 0; v < 7; v++) for (int c = 0; c < C; c++) ft[(size_t)c * nv + v * 11] = 0;   /* zero vectors: p 0.5, unit sum 0 */
    void *dft = nn_malloc(nv * C * 2), *dl = nn_malloc(nv * 4); float *dpr = nn_malloc(C * 4); double *dacc = nn_malloc((2 * C + 1) * 8);
    nn_h2d(dft, ft, nv * C * 2); nn_h2d(dpr, pr, C * 4);
    const float thr = 0.75f;
    for (int h16 = 0; h16 < 2; h16++) {
        /* the seed sums, host reference in double */
        double ra[2 * C + 1]; memset(ra, 0, sizeof ra);
        for (size_t v = 0; v < nv; v++) {
            const int z = z0 + (int)(v / (H * W)), y = (int)(v / W % H), x = (int)(v % W); const float l = h16 ? h2f(lgh[v]) : lgf[v];
            if (z < hz || z >= D - hz || y < hy || y >= H - hy || x < hx || x >= W - hx || !(l >= thr)) continue;
            double s2 = 0; for (int c = 0; c < C; c++) { const double f = h2f(ft[(size_t)c * nv + v]); s2 += f * f; }
            for (int c = 0; c < C; c++) { const double f = h2f(ft[(size_t)c * nv + v]); ra[c] += f; ra[C + c] += s2 > 0 ? f / sqrt(s2) : 0; }
            ra[2 * C] += 1;
        }
        nn_h2d(dl, h16 ? (const void *)lgh : (const void *)lgf, nv * (h16 ? 2 : 4)); nn_zero(dacc, (2 * C + 1) * 8);
        nn_proto_acc(dft, C, z0, nz, D, H, W, hz, hy, hx, dl, h16, thr, dacc);
        double ga[2 * C + 1]; nn_d2h(ga, dacc, (2 * C + 1) * 8); double e = 0;
        for (int i = 0; i < 2 * C; i++) e = fmax(e, fabs(ga[i] - ra[i]) / (1 + fabs(ra[i])));
        CHECK(ga[2 * C] == ra[2 * C] && ra[2 * C] > 20 && e < 1e-5, "h16 %d: seed count %g vs %g, sums max rel err %.3g", h16, ga[2 * C], ra[2 * C], e);
        /* the cosine into the logits */
        nn_proto_cos(dft, C, nv, dpr, dl, h16);
        if (h16) nn_d2h(lgh, dl, nv * 2); else nn_d2h(lgf, dl, nv * 4);
        double pe = 0;
        for (size_t v = 0; v < nv; v++) {
            double d = 0, s2 = 0; for (int c = 0; c < C; c++) { const double f = h2f(ft[(size_t)c * nv + v]); d += f * pr[c]; s2 += f * f; }
            const double p = s2 > 0 ? 0.5 * (1 + d / sqrt(s2)) : 0.5, l = h16 ? h2f(lgh[v]) : lgf[v], pg = 1 / (1 + exp(-l));
            pe = fmax(pe, fabs(pg - p));
        }
        CHECK(pe < 2e-3, "h16 %d: cosine p max err %.3g", h16, pe);
        for (size_t v = 0; v < nv; v++) { const float l = ((int)(rnd() % 129) - 64) / 16.f; lgf[v] = l; lgh[v] = f2h(l); }   /* fresh logits for the next pass */
    }
    const char *err = nn_check(); CHECK(!err, "cuda: %s", err ? err : "");
    nn_free(dft); nn_free(dl); nn_free(dpr); nn_free(dacc); free(ft); free(lgf); free(lgh);
}
int main(void) {
    test_chunk(); test_file();
    if (nn_init(0)) { fprintf(stderr, "test_ink_proto: no GPU\n"); return 1; }
    test_gpu();
    if (fails) { fprintf(stderr, "test_ink_proto: %d FAILED\n", fails); return 1; }
    printf("test_ink_proto: ok\n");
    return 0;
}
