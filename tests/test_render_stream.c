/* The streamed render stage against the in-memory / tiled versions, byte for byte:
   - render_wrap_chunked (chunk-ordered, decoders ahead of renderers, only the volcomp substreams under a block decoded): two zarr volumes (lossy, lossless) (128^3 chunks, a partial edge chunk row, an
     empty region) through a curved grid with invalid vertices and a fold, depth 2 on the first and 0 on the second volume, vs main.c's
     render_rows arithmetic on the whole volume read densely; a tiny budget (decoders wait for the renderers), a large one, --crop;
   - tiffw_write (strips compressed on threads, header patched) read back exactly, at several strip counts;
   - ink-compare streamed (cmd_ink_compare) vs ink_compare_mem: the JPEGs and the statistics (every bit, UFSM_INKCMP_FULL) for the whole
     render (labels, --stretch, --stretch-ct, --max-px), a crop with --each / --each-ct, --fixed with --full and two inks, the 5% skip. */
#include "imgops.h"
#include "store.h"
#include "tiff.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_ink_compare(int argc, char **argv);
int ink_compare_mem(int argc, char **argv);
int cmd_jpeg_fit(int argc, char **argv);

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { *n = 0; return nullptr; }
    fseek(f, 0, SEEK_END); long L = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)L + 1); *n = fread(b, 1, (size_t)L, f); fclose(f); return b;
}
static int same_file(const char *a, const char *b) {
    size_t na, nb; uint8_t *x = slurp(a, &na), *y = slurp(b, &nb);
    const int r = x && y && na == nb && na > 0 && !memcmp(x, y, na); free(x); free(y); return r;
}

/* main.c trilin / render_rows on a dense volume */
static float trilin(const uint8_t *v, const int64_t n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return 0.f;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    if (z0 >= n[0] - 1) z0 = (int)n[0] - 2;
    if (y0 >= n[1] - 1) y0 = (int)n[1] - 2;
    if (x0 >= n[2] - 1) x0 = (int)n[2] - 2;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = z0 + (c >> 2 & 1), b = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        if (a >= n[0] || b >= n[1] || e >= n[2]) continue;
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[((size_t)a * n[1] + b) * n[2] + e];
    }
    return (float)s;
}
static void ref_render(const float *X, const float *Y, const float *Z, int w, int h, double gs, const uint8_t *vol, const int64_t n[3], int depth, uint8_t *img, int ow, int oh) {
    for (int py = 0; py < oh; py++) for (int px = 0; px < ow; px++) {
        const double gu = px / gs, gv = py / gs;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = gu - c0, fv = gv - r0;
        const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
        if (!(Z[i00] >= 0 && Z[i01] >= 0 && Z[i10] >= 0 && Z[i11] >= 0)) continue;
        const float *A[3] = {Z, Y, X}; double P[3], du[3], dv[3];
        for (int d = 0; d < 3; d++) {
            const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
            P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
            du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
        }
        double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
        for (int d = 0; d < 3; d++) N[d] /= nl;
        float m = 0;
        for (int t = -depth; t <= depth; t++) { const float v = trilin(vol, n, P[0] + t * N[0], P[1] + t * N[1], P[2] + t * N[2]); if (v > m) m = v; }
        img[(size_t)py * ow + px] = (uint8_t)lroundf(m);
    }
}

static int run(int (*f)(int, char **), char **argv, const char *out) {   /* argv nullptr-terminated; stdout -> out */
    int argc = 0; while (argv[argc]) argc++;
    fflush(stdout); const int keep = dup(1), fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644); dup2(fd, 1); close(fd);
    const int rc = f(argc, argv);
    fflush(stdout); dup2(keep, 1); close(keep);
    return rc;
}

int main(void) {
    char dir[] = "/tmp/test_render_stream_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char p[800], q[800];
    /* ---- render_wrap_chunked ---- */
    {
        const int S = 128; const int64_t shape[3] = {300, 260, 270};
        uint8_t *buf = malloc((size_t)S * S * S), *dense[2];
        z3 *zz[2];
        for (int v = 0; v < 2; v++) {
            snprintf(p, sizeof p, "%s/v%d.zarr/0", dir, v);
            z3w *w = z3w_create(p, shape, S, v ? 0 : 2, 0, "{}"); CHECK(w != nullptr, "z3w_create %s", z3w_error());   /* lossy (q 2) and coded (q 0) volcomp: the substream decode of both */
            for (int sz = 0; sz < 3; sz++) for (int sy = 0; sy < 3; sy++) for (int sx = 0; sx < 3; sx++) {
                if (v == 1 && sz == 0 && sy == 0) continue;   /* absent shards of the second volume */
                for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
                    const int gz = sz * S + z, gy = sy * S + y, gx = sx * S + x;
                    int val = v ? (gx * 7 + gy * 3 + gz * 5) % 251 : (int)(127 + 120 * sin(gx * 0.11) * cos(gy * 0.07 + gz * 0.05));
                    if (gx > 200 && gy > 200) val = 0;   /* an empty corner */
                    buf[((size_t)z * S + y) * S + x] = (uint8_t)val;
                }
                CHECK(!z3w_write_shard(w, sz, sy, sx, buf, 2), "write shard %s", z3w_error());
            }
            CHECK(!z3w_close(w), "close");
            snprintf(q, sizeof q, "v%d.zarr/0", v);
            zz[v] = z3_open(store_open(dir), q, nullptr); CHECK(zz[v] != nullptr, "z3_open %s", z3_error());
            dense[v] = malloc((size_t)shape[0] * shape[1] * shape[2]);
            const int64_t o[3] = {0, 0, 0}; CHECK(!z3_read(zz[v], o, shape, dense[v], 2), "dense read");
        }
        /* a cylinder section around the z axis: rows along z, columns around, radius 110 +- a ripple; a hole and a strip off the volume */
        const int w = 70, h = 64; float *X = malloc(sizeof(float) * w * h), *Y = malloc(sizeof(float) * w * h), *Z = malloc(sizeof(float) * w * h);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; const double th = c * 0.06, rad = 110 + 6 * sin(r * 0.3) + (c > 50 ? (c - 50) * 2.0 : 0);
            X[i] = (float)(135 + rad * cos(th)); Y[i] = (float)(130 + rad * sin(th)); Z[i] = (float)(10 + r * 4.3);
            if ((r > 20 && r < 26 && c > 10 && c < 18) || (r == 40 && c == 33)) X[i] = Y[i] = Z[i] = -1;
        }
        const double gs = 2.5; const int ow = render_out_px(w, gs), oh = render_out_px(h, gs); const size_t no = (size_t)ow * oh;
        uint8_t *ref[2]; const int dp[2] = {2, 0};
        for (int v = 0; v < 2; v++) { ref[v] = calloc(no, 1); ref_render(X, Y, Z, w, h, gs, dense[v], shape, dp[v], ref[v], ow, oh); }
        size_t nz = 0; for (size_t i = 0; i < no; i++) nz += ref[0][i] != 0;
        CHECK(nz > no / 3, "reference render mostly empty (%zu of %zu)", nz, no);
        const size_t budgets[3] = {1, (size_t)3 << 21, (size_t)1 << 32};
        for (int bi = 0; bi < 3; bi++) for (int crop = 0; crop < 2; crop++) {
            uint8_t *im[2] = {calloc(no, 1), calloc(no, 1)}; const int cr[4] = {37, 51, 60, 44};
            CHECK(!render_wrap_chunked(zz, 2, dp, X, Y, Z, w, h, gs, im, ow, oh, crop ? cr : nullptr, budgets[bi], 3, 2, 0), "render_wrap_chunked");
            for (int v = 0; v < 2; v++) {
                size_t bad = 0;
                for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
                    const size_t i = (size_t)y * ow + x;
                    if (crop && (x < cr[0] || x >= cr[0] + cr[2] || y < cr[1] || y >= cr[1] + cr[3])) continue;
                    bad += im[v][i] != ref[v][i];
                }
                CHECK(!bad, "budget %zu crop %d volume %d: %zu pixels differ", budgets[bi], crop, v, bad);
            }
            free(im[0]); free(im[1]);
        }
        /* one volume */
        { uint8_t *im[1] = {calloc(no, 1)}; z3 *z1[1] = {zz[1]}; const int d1[1] = {1}; uint8_t *r1 = calloc(no, 1);
          ref_render(X, Y, Z, w, h, gs, dense[1], shape, 1, r1, ow, oh);
          CHECK(!render_wrap_chunked(z1, 1, d1, X, Y, Z, w, h, gs, im, ow, oh, nullptr, 1, 2, 4, 0) && !memcmp(im[0], r1, no), "one volume");
          free(im[0]); free(r1); }
        /* render_wraps_chunked: this grid and a second one 3 voxels outward (with its own holes) in one pass = each grid alone */
        {
            float *X2 = malloc(sizeof(float) * w * h), *Y2 = malloc(sizeof(float) * w * h), *Z2 = malloc(sizeof(float) * w * h);
            for (size_t i = 0; i < (size_t)w * h; i++) {
                const double dx = X[i] - 135, dy = Y[i] - 130, rr = hypot(dx, dy);
                X2[i] = Z[i] < 0 ? -1 : (float)(135 + dx * (rr + 3) / rr); Y2[i] = Z[i] < 0 ? -1 : (float)(130 + dy * (rr + 3) / rr); Z2[i] = Z[i];
                if (i % 97 == 5) X2[i] = Y2[i] = Z2[i] = -1;
            }
            uint8_t *r2[2]; for (int v = 0; v < 2; v++) { r2[v] = calloc(no, 1); ref_render(X2, Y2, Z2, w, h, gs, dense[v], shape, dp[v], r2[v], ow, oh); }
            const float *const XS[2] = {X, X2}, *const YS[2] = {Y, Y2}, *const ZS[2] = {Z, Z2};
            for (int bi = 0; bi < 3; bi += 2) {
                uint8_t *im[4] = {calloc(no, 1), calloc(no, 1), calloc(no, 1), calloc(no, 1)};
                CHECK(!render_wraps_chunked(zz, 2, dp, 2, XS, YS, ZS, w, h, gs, im, ow, oh, nullptr, budgets[bi], 3, 2, 0), "render_wraps_chunked");
                CHECK(!memcmp(im[0], ref[0], no) && !memcmp(im[1], ref[1], no), "joint budget %zu: grid 0 differs", budgets[bi]);
                CHECK(!memcmp(im[2], r2[0], no) && !memcmp(im[3], r2[1], no), "joint budget %zu: grid 1 differs", budgets[bi]);
                for (int k = 0; k < 4; k++) free(im[k]);
            }
            free(r2[0]); free(r2[1]); free(X2); free(Y2); free(Z2);
        }
        for (int v = 0; v < 2; v++) { z3_close(zz[v]); free(dense[v]); free(ref[v]); }
        free(X); free(Y); free(Z); free(buf);
    }
    /* ---- tiffw_write, threaded: read back ---- */
    const int W = 1503, H = 1301; uint8_t *ct = malloc((size_t)W * H), *ink = malloc((size_t)W * H), *ink2 = malloc((size_t)W * H);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = (size_t)y * W + x; const double rr = hypot(x - 700.0, (y - 640.0) * 1.3);
        ct[i] = rr < 600 ? (uint8_t)(60 + (x * 13 + y * 7) % 150) : 0;
        ink[i] = (uint8_t)(rr < 600 ? fmin(255, 30 + 100 * (1 + sin(x * 0.05) * sin(y * 0.031)) + (x * y) % 7) : (x + y) % 11);
        ink2[i] = (uint8_t)((ink[i] * 3 + ct[i]) / 4);
    }
    for (int hh = 1; hh <= H; hh += 650) {   /* 1, 651 (one partial strip), 1301 rows */
        snprintf(p, sizeof p, "%s/t.tif", dir); CHECK(!tiffw_write(p, W, hh, 1, ink, 1), "tiffw_write");
        int w2, h2; uint8_t *b = img_read_u8(p, &w2, &h2);
        CHECK(b && w2 == W && h2 == hh && !memcmp(b, ink, (size_t)W * hh), "tiffw round trip %d rows", hh); free(b);
    }
    /* ---- ink-compare: streamed vs in memory ---- */
    char cp[700], ip[700], ip2[700], ia[760], ib[760];
    snprintf(cp, sizeof cp, "%s/w_ct.tif", dir); snprintf(ip, sizeof ip, "%s/w.tif", dir); snprintf(ip2, sizeof ip2, "%s/w2.tif", dir);
    CHECK(!tiffw_write(cp, W, H, 1, ct, 1) && !tiffw_write(ip, W, H, 1, ink, 1) && !tiffw_write(ip2, W, H, 1, ink2, 1), "write renders");
    snprintf(ia, sizeof ia, "gamb=%s", ip); snprintf(ib, sizeof ib, "other=%s", ip2);
    setenv("UFSM_INKCMP_FULL", "1", 1);
    for (int cs = 0; cs < 4; cs++) {
        char o[2][760], s[2][760], e[2][760];
        for (int m = 0; m < 2; m++) {
            snprintf(o[m], sizeof o[m], "%s/c%d_%d.jpg", dir, cs, m); snprintf(s[m], sizeof s[m], "%s/c%d_%d.txt", dir, cs, m); snprintf(e[m], sizeof e[m], "%s/e%d_%d_", dir, cs, m);
            char *a0[] = {"ufsm", "ink-compare", o[m], "--ct", cp, "--ink", ia, "--px-um", "4.8", "--stretch", "--stretch-ct", "--max-px", "700", "--threads", "3", nullptr};
            char *a1[] = {"ufsm", "ink-compare", "-", "--ct", cp, "--ink", ia, "--px-um", "4.8", "--stretch", "--stretch-ct", "--crop", "301,257,801,603", "--each", e[m], "--each-ct", "--threads", "4", nullptr};
            char *a2[] = {"ufsm", "ink-compare", o[m], "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "2.4", "--fixed", "20,180", "--full", "--crop", "100,90,700,500", "--threads", "2", nullptr};
            char *a3[] = {"ufsm", "ink-compare", o[m], "--ct", cp, "--ink", ia, "--crop", "0,0,120,100", nullptr};   /* under 5% CT */
            char **a = cs == 0 ? a0 : cs == 1 ? a1 : cs == 2 ? a2 : a3;
            const int rc = run(m ? ink_compare_mem : cmd_ink_compare, a, s[m]);
            CHECK(rc == (cs == 3 ? 3 : 0), "case %d mode %d rc %d", cs, m, rc);
        }
        CHECK(same_file(s[0], s[1]), "case %d: statistics differ", cs);
        if (cs == 0 || cs == 2) CHECK(same_file(o[0], o[1]), "case %d: JPEG differs", cs);
        if (cs == 1) for (int k = 0; k < 2; k++) {
            char x[800], y[800]; snprintf(x, sizeof x, "%s%s.jpg", e[0], k ? "gamb" : "ct"); snprintf(y, sizeof y, "%s%s.jpg", e[1], k ? "gamb" : "ct");
            CHECK(same_file(x, y), "case 1: %s differs", x);
        }
    }
    {   /* one call, several outputs (--then): the bytes of separate calls (rgb-agree full + fitted, panels without CT, --each fitted) */
        char o1[760], o2[760], o3[760], e1[760], sv[760], r1[760], r2[760], r3[760], f1[760], g1[760];
        snprintf(o1, sizeof o1, "%s/m1.jpg", dir); snprintf(o2, sizeof o2, "%s/m2.jpg", dir); snprintf(o3, sizeof o3, "%s/m3.jpg", dir); snprintf(e1, sizeof e1, "%s/me_", dir);
        snprintf(r1, sizeof r1, "%s/s1.jpg", dir); snprintf(r2, sizeof r2, "%s/s2.jpg", dir); snprintf(r3, sizeof r3, "%s/s3.jpg", dir); snprintf(f1, sizeof f1, "%s/se_", dir);
        snprintf(sv, sizeof sv, "%s/m.txt", dir);
        char *m[] = {"ufsm", "ink-compare", o1, "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "4.8", "--stretch", "--stretch-ct", "--threads", "3", "--rgb-agree", "gamb,other",
                     "--then", o2, "--rgb-agree", "gamb,other", "--max-px", "500",
                     "--then", o3, "--cmap", "viridis", "--no-ct", "--max-px", "640",
                     "--then", "-", "--cmap", "viridis", "--each", e1, "--each-suffix", "_v", "--max-px", "300", nullptr};
        CHECK(run(cmd_ink_compare, m, sv) == 0, "--then rc");
        char *s1[] = {"ufsm", "ink-compare", r1, "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "4.8", "--stretch", "--stretch-ct", "--threads", "2", "--rgb-agree", "gamb,other", nullptr};
        char *s2[] = {"ufsm", "ink-compare", r2, "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "4.8", "--stretch", "--stretch-ct", "--threads", "2", "--rgb-agree", "gamb,other", "--max-px", "500", nullptr};
        char *s3[] = {"ufsm", "ink-compare", r3, "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "4.8", "--stretch", "--stretch-ct", "--threads", "2", "--cmap", "viridis", "--no-ct", "--max-px", "640", nullptr};
        char *s4[] = {"ufsm", "ink-compare", "-", "--ct", cp, "--ink", ia, "--ink", ib, "--px-um", "4.8", "--stretch", "--stretch-ct", "--threads", "2", "--cmap", "viridis", "--each", f1, "--each-suffix", "_v", "--max-px", "300", nullptr};
        snprintf(g1, sizeof g1, "%s/s.txt", dir);
        CHECK(run(cmd_ink_compare, s1, g1) == 0 && run(cmd_ink_compare, s2, g1) == 0 && run(cmd_ink_compare, s3, g1) == 0 && run(cmd_ink_compare, s4, g1) == 0, "separate rc");
        CHECK(same_file(sv, g1), "--then: statistics differ");
        CHECK(same_file(o1, r1) && same_file(o2, r2) && same_file(o3, r3), "--then: JPEG differs");
        for (int k = 0; k < 2; k++) { char x[800], y[800]; snprintf(x, sizeof x, "%s%s_v.jpg", e1, k ? "other" : "gamb"); snprintf(y, sizeof y, "%s%s_v.jpg", f1, k ? "other" : "gamb");
                                      CHECK(same_file(x, y), "--then: %s differs", x); }
    }
    {   /* --diff-agree: an ink against itself is viridis 0 on papyrus; against another ink, brighter */
        char o[760], ic[760], idd[760], sv[760]; snprintf(o, sizeof o, "%s/d.jpg", dir); snprintf(sv, sizeof sv, "%s/d.txt", dir);
        snprintf(ic, sizeof ic, "copy=%s", ip); snprintf(idd, sizeof idd, "%s", ib);
        double mean[2] = {0, 0};
        for (int c = 0; c < 2; c++) {
            char *a[] = {"ufsm", "ink-compare", o, "--ct", cp, "--ink", ia, "--ink", c ? idd : ic, "--diff-agree", c ? "gamb,other" : "gamb,copy", "--full", "--threads", "2", nullptr};
            CHECK(run(cmd_ink_compare, a, sv) == 0, "diff-agree %d rc", c);
            int w = 0, h = 0; uint8_t *j = jpeg_read_rgb(o, &w, &h); long nn = 0, bad = 0;
            CHECK(j && w == W && h == H, "diff-agree %d size %dx%d", c, w, h);
            for (int y = h / 4; j && y < h; y++) for (int x = 0; x < w; x++) {
                const uint8_t *q = j + ((size_t)y * w + x) * 3; if (abs(q[0] - 40) < 6 && abs(q[1] - 40) < 6 && abs(q[2] - 40) < 6) continue;   /* off papyrus */
                nn++; mean[c] += q[1]; if (!c && (abs(q[0] - 68) > 12 || q[1] > 14 || abs(q[2] - 84) > 12)) bad++;
            }
            if (nn) mean[c] /= nn;
            CHECK(nn > 0 && bad < nn / 100, "diff-agree %d: %ld of %ld papyrus pixels off viridis 0", c, bad, nn);
            free(j);
        }
        CHECK(mean[1] > mean[0] + 3, "diff-agree: disagreement not brighter (%.1f vs %.1f)", mean[1], mean[0]);
    }
    {   /* --papyrus: papyrus is warm (R > G > B); the ink darkens it */
        char o[760], sv[760]; snprintf(o, sizeof o, "%s/p.jpg", dir); snprintf(sv, sizeof sv, "%s/p.txt", dir);
        char *a[] = {"ufsm", "ink-compare", o, "--ct", cp, "--ink", ia, "--papyrus", "gamb", "--stretch", "--stretch-ct", "--full", "--threads", "2", nullptr};
        CHECK(run(cmd_ink_compare, a, sv) == 0, "papyrus rc");
        int w = 0, h = 0; uint8_t *j = jpeg_read_rgb(o, &w, &h); long warm = 0, nn = 0; double lum[2] = {0, 0}; long cnt[2] = {0, 0};
        CHECK(j && w == W && h == H, "papyrus size %dx%d", w, h);
        for (int y = 0; j && y < h; y++) for (int x = 0; x < w; x++) {
            const uint8_t *q = j + ((size_t)y * w + x) * 3; if (q[0] < 24 && q[1] < 24 && q[2] < 24) continue;
            nn++; if (q[0] >= q[1] && q[1] >= q[2]) warm++;
            const int k = ink[(size_t)y * W + x] > 128; lum[k] += q[0] + q[1] + q[2]; cnt[k]++;
        }
        CHECK(nn > 0 && warm > nn * 9 / 10, "papyrus: %ld of %ld pixels warm", warm, nn);
        {   /* --papyrus-floor median: no papyrus pixel darker than the floor's colour, so the mean brightens */
            char o2[760]; snprintf(o2, sizeof o2, "%s/p2.jpg", dir);
            char *b[] = {"ufsm", "ink-compare", o2, "--ct", cp, "--ink", ia, "--papyrus", "gamb", "--papyrus-floor", "median", "--stretch", "--stretch-ct", "--full", "--threads", "2", nullptr};
            CHECK(run(cmd_ink_compare, b, sv) == 0, "papyrus floor rc");
            int w2 = 0, h2 = 0; uint8_t *k = jpeg_read_rgb(o2, &w2, &h2); double s0 = 0, s1 = 0;
            for (size_t i = 0; j && k && w2 == w && h2 == h && i < (size_t)w * h * 3; i++) { s0 += j[i]; s1 += k[i]; }
            CHECK(k && s1 > s0, "papyrus floor: not brighter (%.0f vs %.0f)", s1, s0);
            free(k);
        }
        if (cnt[0] && cnt[1]) CHECK(lum[1] / cnt[1] < lum[0] / cnt[0], "papyrus: inked pixels not darker");
        free(j);
    }
    {   /* jpeg-fit --div 2 in place: half per side, mean colour kept */
        char o[760], sv[760]; snprintf(o, sizeof o, "%s/d.jpg", dir); snprintf(sv, sizeof sv, "%s/f.txt", dir);
        int w0, h0; uint8_t *a = jpeg_read_rgb(o, &w0, &h0); double m0 = 0, m1 = 0;
        for (size_t i = 0; a && i < (size_t)w0 * h0 * 3; i++) m0 += a[i];
        char *f[] = {"ufsm", "jpeg-fit", o, o, "--div", "2", "--threads", "2", nullptr};
        CHECK(run(cmd_jpeg_fit, f, sv) == 0, "jpeg-fit rc");
        int w1, h1; uint8_t *b = jpeg_read_rgb(o, &w1, &h1);
        CHECK(b && abs(w1 - w0 / 2) <= 1 && abs(h1 - h0 / 2) <= 1, "jpeg-fit %dx%d -> %dx%d", w0, h0, w1, h1);
        for (size_t i = 0; b && i < (size_t)w1 * h1 * 3; i++) m1 += b[i];
        if (a && b) CHECK(fabs(m0 / ((double)w0 * h0 * 3) - m1 / ((double)w1 * h1 * 3)) < 2.0, "jpeg-fit mean moved");
        char *g[] = {"ufsm", "jpeg-fit", o, o, nullptr};
        CHECK(run(cmd_jpeg_fit, g, sv) == 2, "jpeg-fit without --div / --max-px");
        free(a); free(b);
    }
    free(ct); free(ink); free(ink2);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {}
    if (fails) { fprintf(stderr, "test_render_stream: %d failures\n", fails); return 1; }
    printf("test_render_stream: OK\n");
    return 0;
}
