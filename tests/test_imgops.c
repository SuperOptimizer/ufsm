/* src/imgops.c: Pillow-exact BILINEAR / BOX resampling and the overlay_red.py compositing (bytes from Pillow / numpy, see
   test_imgops_expected.h), the L1 dilation against a brute-force cross dilation, a JPEG write / PNG read smoke test, and the
   tifxyz rescale of xform_wraps.py (float32 v / div - offset, -1 kept, meta scale x div). */
#include "imgops.h"
#include "tiffw.h"
#include "test_imgops_expected.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void lcg(uint8_t *o, int n, int seed) { unsigned s = (unsigned)seed; for (int i = 0; i < n; i++) { s = (s * 1103515245u + 12345u) & 0x7fffffffu; o[i] = (uint8_t)((s >> 16) & 255); } }

static void test_resample(void) {
    uint8_t a[35], b[117], c[231], d[45];
    lcg(a, 35, 1); CHECK(!pil_resize(a, 7, 5, 1, b, 13, 9, PIL_BILINEAR, 2), "bilinear resize");
    CHECK(!memcmp(b, exp_bilinear, sizeof b), "bilinear 7x5 -> 13x9 differs from Pillow");
    lcg(c, 231, 2); CHECK(!pil_resize(c, 11, 7, 3, d, 5, 3, PIL_BOX, 3), "box resize");
    CHECK(!memcmp(d, exp_box, sizeof d), "box RGB 11x7 -> 5x3 differs from Pillow");
}
static void test_overlay(void) {
    enum { W = 61, H = 45, IW = 31, IH = 23 };
    uint8_t ct[W * H], ink[3][IW * IH];
    lcg(ct, W * H, 3);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if ((y < 10 && x < 12) || (y >= 30 && x >= 50)) ct[y * W + x] = 0;
    for (int k = 0; k < 3; k++) lcg(ink[k], IW * IH, 4 + k);
    const uint8_t *ip[3] = {ink[0], ink[1], ink[2]}; int ow, oh;
    uint8_t *o = overlay_red_half(ct, W, H, ip, 3, IW, IH, &ow, &oh, 3);
    CHECK(o && ow == EXP_OW && oh == EXP_OH, "overlay size %d x %d", ow, oh);
    if (o && ow == EXP_OW && oh == EXP_OH) {
        int mx = 0; for (size_t i = 0; i < sizeof exp_overlay; i++) { const int e = abs(o[i] - exp_overlay[i]); if (e > mx) mx = e; }
        CHECK(mx == 0, "overlay differs from overlay_red.py by up to %d", mx);
    }
    free(o);
    CHECK(!overlay_red_skip(ct, W * H), "skip rule: a mostly filled CT kept");
    uint8_t z[100] = {0}; z[0] = z[1] = z[2] = z[3] = 1; CHECK(overlay_red_skip(z, 100), "skip rule: 4%% data skipped");
    z[4] = 1; CHECK(!overlay_red_skip(z, 100), "skip rule: 5%% data kept");
}
static void test_dilate(void) {
    enum { W = 53, H = 41 }; uint8_t ct[W * H], m[W * H];
    lcg(ct, W * H, 9); for (int i = 0; i < W * H; i++) ct[i] = ct[i] < 3 ? 0 : ct[i];
    l1_dilate_zero(ct, W, H, 7, m); int bad = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int e = 0; for (int v = 0; v < H && !e; v++) for (int u = 0; u < W; u++) if (!ct[v * W + u] && abs(v - y) + abs(u - x) <= 7) { e = 1; break; }
        bad += e != m[y * W + x];
    }
    CHECK(!bad, "L1 dilation: %d pixels differ from the brute-force cross dilation", bad);
}
static void test_jpeg(void) {
    enum { W = 40, H = 24 }; uint8_t rgb[W * H * 3]; for (int i = 0; i < W * H * 3; i++) rgb[i] = (uint8_t)(i * 7);
    const char *p = "/tmp/ufsm_test_imgops.jpg"; CHECK(!jpeg_write_rgb(p, W, H, rgb, 90), "jpeg write");
    FILE *f = fopen(p, "rb"); uint8_t h[3] = {0}; if (f) { CHECK(fread(h, 1, 3, f) == 3, "jpeg read"); fclose(f); }
    CHECK(h[0] == 0xff && h[1] == 0xd8 && h[2] == 0xff, "jpeg SOI marker"); remove(p);
}
static void test_tifxyz_scale(void) {
    enum { W = 9, H = 5 }; float x[W * H], y[W * H], z[W * H];
    for (int i = 0; i < W * H; i++) { x[i] = 17761.5f + i * 3.37f; y[i] = 6691.428f + i * 1.11f; z[i] = i % 7 == 3 ? -1.f : 9217.504f + i * 0.31f; }
    const char *s = "/tmp/ufsm_test_imgops_src", *d = "/tmp/ufsm_test_imgops_dst"; char p[256];
    CHECK(!tifxyz_write(s, W, H, x, y, z, 4, 2.399, 150, "test"), "tifxyz write");
    const double off[3] = {10, 20, 30};   /* z, y, x */
    CHECK(!tifxyz_scale(s, d, 4, off), "tifxyz scale");
    int w, h; snprintf(p, sizeof p, "%s/x.tif", d); float *X = img_read_f32(p, &w, &h);
    snprintf(p, sizeof p, "%s/z.tif", d); float *Z = img_read_f32(p, &w, &h);
    int bad = 0;
    for (int i = 0; X && Z && i < W * H; i++) {
        const float ex = z[i] < 0 ? -1.f : x[i] / 4.f - 30.f, ez = z[i] < 0 ? -1.f : z[i] / 4.f - 10.f;
        bad += memcmp(&X[i], &ex, 4) != 0 || memcmp(&Z[i], &ez, 4) != 0;
    }
    CHECK(X && Z && !bad, "tifxyz scale: %d vertices differ", bad);
    snprintf(p, sizeof p, "%s/meta.json", d); FILE *f = fopen(p, "rb"); char m[2048] = {0};
    if (f) { CHECK(fread(m, 1, sizeof m - 1, f) > 0, "meta read"); fclose(f); }
    CHECK(strstr(m, "\"scale\": [1, 1]") && strstr(m, "\"voxel_size_um\": 9.596") && strstr(m, "\"winding\": 150"), "tifxyz scale meta: %s", m);
    free(X); free(Z);
    snprintf(p, sizeof p, "rm -rf %s %s", s, d); CHECK(!system(p), "cleanup");
}
int main(void) {
    test_resample(); test_overlay(); test_dilate(); test_jpeg(); test_tifxyz_scale();
    if (fails) { fprintf(stderr, "test_imgops: %d failures\n", fails); return 1; }
    printf("test_imgops: ok\n");
    return 0;
}
