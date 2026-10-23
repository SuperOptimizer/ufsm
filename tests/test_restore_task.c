/* task restore (src/restore.h, src/nn_restore.cu, train --task restore, predict on a restore checkpoint):
   codec round trip vs volcomp, pyramid / scanner degradation vs CPU references, the split input, loss + gradient vs a double reference and
   central differences, identity at step 0 (zero head), the sampler's clean-window pairing, and the predict restore path round trip */
#include "nn.h"
#include "restore.h"
#include "srpair.h"
#include "sample.h"
#include "sources.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include "volcomp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
/* a CT-like volume: sheets (sinusoidal bands) + noise, zero outside a sphere */
static void ct_like(uint8_t *v, int D, int H, int W, int zero_out) {
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const double r = sqrt((z - D / 2.) * (z - D / 2.) + (y - H / 2.) * (y - H / 2.) + (x - W / 2.) * (x - W / 2.));
        double val = 90 + 60 * sin(0.45 * x + 0.2 * y) * cos(0.13 * z) + 12 * (urand() - 0.5);
        if (zero_out && r > 0.45 * (D < W ? D : W)) val = 0;
        else val = val < 1 ? 1 : val > 255 ? 255 : val;
        v[((size_t)z * H + y) * W + x] = (uint8_t)lrint(val);
    }
}
/* trilinear as the GPU does it (restore_up_k), masked by m (or the nearest coarse voxel when m is null) */
static void up_ref(const uint8_t *s, int sd, int sh, int sw, uint8_t *o, int D, int H, int W, const float st[3], const float ph[3], const uint8_t *m) {
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const float cz = ((float)z + .5f) / st[0] - .5f + ph[0], cy = ((float)y + .5f) / st[1] - .5f + ph[1], cx = ((float)x + .5f) / st[2] - .5f + ph[2];
        const int iz = (int)floorf(cz), iy = (int)floorf(cy), ix = (int)floorf(cx); const float tz = cz - iz, ty = cy - iy, tx = cx - ix;
        if (m) { if (!m[i]) { o[i] = 0; continue; } }
        else { int nz = (int)floorf(cz + .5f), ny = (int)floorf(cy + .5f), nx = (int)floorf(cx + .5f); nz = nz < 0 ? 0 : nz >= sd ? sd - 1 : nz; ny = ny < 0 ? 0 : ny >= sh ? sh - 1 : ny; nx = nx < 0 ? 0 : nx >= sw ? sw - 1 : nx;
               if (!s[((size_t)nz * sh + ny) * sw + nx]) { o[i] = 0; continue; } }
        float a = 0;
        for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int A = iz + dz, Bq = iy + dy, C = ix + dx; A = A < 0 ? 0 : A >= sd ? sd - 1 : A; Bq = Bq < 0 ? 0 : Bq >= sh ? sh - 1 : Bq; C = C < 0 ? 0 : C >= sw ? sw - 1 : C;
            a += (dz ? tz : 1 - tz) * (dy ? ty : 1 - ty) * (dx ? tx : 1 - tx) * s[((size_t)A * sh + Bq) * sw + C];
        }
        const float r = rintf(a); o[i] = (uint8_t)(r < 1 ? 1 : r > 255 ? 255 : r);
    }
}
static int near_equal(const uint8_t *a, const uint8_t *b, size_t n, int tol, double frac_exact, const char *what) {
    size_t neq = 0; int mx = 0;
    for (size_t i = 0; i < n; i++) { const int d = abs((int)a[i] - (int)b[i]); if (d) neq++; if (d > mx) mx = d; }
    const int ok = mx <= tol && (double)neq <= (1 - frac_exact) * (double)n;
    if (!ok) fprintf(stderr, "  %s: max |diff| %d, %zu of %zu differ\n", what, mx, neq, n);
    return ok;
}

static void test_codec(void) {
    enum { C = 128 }; const size_t n = (size_t)C * C * C;
    uint8_t *v = malloc(n), *cpu = malloc(n), *gpu = malloc(n), *enc = malloc(VOLCOMP_ENCODE_BOUND);
    ct_like(v, C, C, C, 1);
    for (int qi = 0; qi < 2; qi++) {
        const float q = qi ? 8.f : 2.f;
        size_t bytes = 0;
        CHECK(!volcomp_encode(v, q, enc, VOLCOMP_ENCODE_BOUND, &bytes) && !volcomp_decode(enc, bytes, cpu, n), "volcomp q %g", q);
        uint8_t *d = nn_malloc(n); nn_h2d(d, v, n); const int ph[3] = {0, 0, 0};
        nn_codec_roundtrip(d, C, C, C, q, ph); nn_d2h(gpu, d, n); nn_free(d);
        CHECK(near_equal(cpu, gpu, n, 1, 0.999, "codec"), "GPU codec round trip != volcomp (q %g)", q);
        double e = 0; for (size_t i = 0; i < n; i++) e += ((double)gpu[i] - v[i]) * ((double)gpu[i] - v[i]);
        CHECK(sqrt(e / n) > 0.3 && sqrt(e / n) < 8, "codec q %g rmse %g implausible", q, sqrt(e / n));
    }
    /* phase: a crop with blocks starting at -phase equals the aligned round trip of the larger volume on the crop */
    {
        const int ph[3] = {3, 5, 7}, d0 = ph[0], d1 = ph[1], d2 = ph[2], cd = 70, chh = 61, cw = 53;   /* crop coordinate -phase = big coordinate 0 */
        uint8_t *crop = malloc((size_t)cd * chh * cw), *ref = malloc(n), *out = malloc((size_t)cd * chh * cw);
        /* the reference: the 128^3 round trip (blocks aligned at 0); the crop starts at d0, d1, d2 so its blocks start at -phase */
        uint8_t *dv = nn_malloc(n); nn_h2d(dv, v, n); const int z0[3] = {0, 0, 0}; nn_codec_roundtrip(dv, C, C, C, 8.f, z0); nn_d2h(ref, dv, n); nn_free(dv);
        for (int z = 0; z < cd; z++) for (int y = 0; y < chh; y++) memcpy(crop + ((size_t)z * chh + y) * cw, v + ((size_t)(z + d0) * C + y + d1) * C + d2, cw);
        uint8_t *dc = nn_malloc((size_t)cd * chh * cw); nn_h2d(dc, crop, (size_t)cd * chh * cw); nn_codec_roundtrip(dc, cd, chh, cw, 8.f, ph); nn_d2h(out, dc, (size_t)cd * chh * cw); nn_free(dc);
        /* interior blocks (fully inside the crop) must agree */
        size_t bad = 0, tot = 0;
        for (int z = 0; z < cd; z++) for (int y = 0; y < chh; y++) for (int x = 0; x < cw; x++) {
            const int bz = (z + ph[0]) / 16 * 16 - ph[0], by = (y + ph[1]) / 16 * 16 - ph[1], bx = (x + ph[2]) / 16 * 16 - ph[2];
            if (bz < 0 || by < 0 || bx < 0 || bz + 16 > cd || by + 16 > chh || bx + 16 > cw) continue;
            tot++; bad += abs((int)out[((size_t)z * chh + y) * cw + x] - (int)ref[((size_t)(z + d0) * C + y + d1) * C + x + d2]) > 1;
        }
        CHECK(tot > 0 && bad == 0, "codec phase: %zu of %zu interior voxels differ", bad, tot);
        free(crop); free(ref); free(out);
    }
    free(v); free(cpu); free(gpu); free(enc);
}

static void test_degrade(void) {
    enum { D = 64, H = 48, W = 80 }; const size_t n = (size_t)D * H * W;
    uint8_t *c = malloc(n), *low = malloc(n), *ref = malloc(n), *co = malloc(n);
    ct_like(c, D, H, W, 1);
    restore_ranges g; restore_ranges_default(&g);
    const size_t sb = restore_scratch_bytes(D, H, W, 3.f);
    uint8_t *dc = nn_malloc(n), *dl = nn_malloc(n), *dco = nn_malloc(n); void *scr = nn_malloc(sb);
    nn_h2d(dc, c, n);
    /* identity */
    restore_recipe r; restore_recipe_draw(&r, RF_IDENTITY, 1, D, H, W, &g);
    CHECK(!nn_restore_degrade(dc, dl, D, H, W, &r, scr, sb, nullptr), "identity"); nn_d2h(low, dl, n);
    CHECK(!memcmp(low, c, n), "identity input != clean");
    /* pyramid without the codec: the exact 4x block mean, rounded, trilinear back, masked by the clean window */
    restore_recipe_draw(&r, RF_PYRAMID, 2, D, H, W, &g); r.q = 0;
    CHECK(r.small[0] == D / 4 && r.small[1] == H / 4 && r.small[2] == W / 4 && r.box, "pyramid recipe geometry");
    CHECK(!nn_restore_degrade(dc, dl, D, H, W, &r, scr, sb, dco), "pyramid"); nn_d2h(low, dl, n);
    const int sd = D / 4, sh = H / 4, sw = W / 4; uint8_t *s = malloc((size_t)sd * sh * sw), *sg = malloc((size_t)sd * sh * sw);
    nn_d2h(sg, dco, (size_t)sd * sh * sw);
    for (int z = 0; z < sd; z++) for (int y = 0; y < sh; y++) for (int x = 0; x < sw; x++) {
        double a = 0; for (int k = 0; k < 64; k++) a += c[((size_t)(4 * z + k / 16) * H + 4 * y + (k / 4) % 4) * W + 4 * x + k % 4];
        a /= 64; s[((size_t)z * sh + y) * sw + x] = a <= 0 ? 0 : (uint8_t)fmin(255, fmax(1, rint(a)));
    }
    CHECK(near_equal(s, sg, (size_t)sd * sh * sw, 1, 0.999, "pyramid coarse"), "pyramid coarse grid != 4x block mean");
    const float st[3] = {4, 4, 4}, ph[3] = {0, 0, 0};
    up_ref(s, sd, sh, sw, ref, D, H, W, st, ph, c);
    CHECK(near_equal(low, ref, n, 1, 0.999, "pyramid"), "pyramid input != CPU reference");
    { size_t bad = 0; for (size_t i = 0; i < n; i++) bad += (c[i] == 0) != (low[i] == 0); CHECK(!bad, "pyramid mask: %zu voxels", bad); }
    /* scanner without noise / tone / codec: the Gaussian PSF sampled on the coarse grid, against a double reference */
    restore_recipe_draw(&r, RF_SCANNER, 3, D, H, W, &g);
    r.noise = r.sig_var = r.corr = r.edge = r.bias = 0; r.gain = r.gamma = 1; r.q = 0;
    CHECK(!nn_restore_degrade(dc, dl, D, H, W, &r, scr, sb, dco), "scanner");
    const int td = r.small[0], th = r.small[1], tw = r.small[2]; uint8_t *sc = malloc((size_t)td * th * tw), *scr_ref = malloc((size_t)td * th * tw);
    nn_d2h(sc, dco, (size_t)td * th * tw);
    double *tmp1 = malloc(sizeof(double) * D * H * tw), *tmp2 = malloc(sizeof(double) * D * th * tw);
    const int dims[3] = {D, H, W}, sm[3] = {td, th, tw};
    double wts[3][128][64]; int j0s[3][128], nts[3][128];
    for (int a = 0; a < 3; a++) for (int i = 0; i < sm[a]; i++) {
        const double xc = (i + 0.5 - r.phase[a]) * r.step[a] - 0.5, sg2 = r.psf[a] < 0.3 ? 0.3 : r.psf[a]; const int rr = (int)ceil(4 * sg2), jc = (int)floor(xc);
        j0s[a][i] = jc - rr; nts[a][i] = 2 * rr + 2; double ws = 0;
        for (int k = 0; k < nts[a][i]; k++) { const double d = (jc - rr + k) - xc; wts[a][i][k] = exp(-d * d / (2 * sg2 * sg2)); ws += wts[a][i][k]; }
        for (int k = 0; k < nts[a][i]; k++) wts[a][i][k] /= ws;
    }
    #define CL(j, N) ((j) < 0 ? 0 : (j) >= (N) ? (N) - 1 : (j))
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int i = 0; i < tw; i++) { double a = 0; for (int k = 0; k < nts[2][i]; k++) a += wts[2][i][k] * c[((size_t)z * H + y) * W + CL(j0s[2][i] + k, W)]; tmp1[((size_t)z * H + y) * tw + i] = a; }
    for (int z = 0; z < D; z++) for (int i = 0; i < th; i++) for (int x = 0; x < tw; x++) { double a = 0; for (int k = 0; k < nts[1][i]; k++) a += wts[1][i][k] * tmp1[((size_t)z * H + CL(j0s[1][i] + k, H)) * tw + x]; tmp2[((size_t)z * th + i) * tw + x] = a; }
    for (int i = 0; i < td; i++) for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) { double a = 0; for (int k = 0; k < nts[0][i]; k++) a += wts[0][i][k] * tmp2[((size_t)CL(j0s[0][i] + k, D) * th + y) * tw + x];
        scr_ref[((size_t)i * th + y) * tw + x] = a <= 0 ? 0 : (uint8_t)fmin(255, fmax(1, rint(a))); }
    CHECK(near_equal(sc, scr_ref, (size_t)td * th * tw, 1, 0.99, "scanner coarse"), "scanner coarse grid != Gaussian reference (step %g psf %g)", r.step[0], r.psf[0]);
    /* the full scanner recipe: noise changes the input, the mask is kept */
    restore_recipe_draw(&r, RF_SCANNER, 4, D, H, W, &g); r.noise = 3;
    CHECK(!nn_restore_degrade(dc, dl, D, H, W, &r, scr, sb, nullptr), "scanner full"); nn_d2h(low, dl, n);
    { size_t bad = 0; for (size_t i = 0; i < n; i++) bad += (c[i] == 0) != (low[i] == 0); CHECK(!bad, "scanner mask: %zu voxels", bad); }
    nn_free(dc); nn_free(dl); nn_free(dco); nn_free(scr);
    free(c); free(low); free(ref); free(co); free(s); free(sg); free(sc); free(scr_ref); free(tmp1); free(tmp2);
}

/* e4m3: exactly representable (|x| a 4-bit significand times a power of two in range) */
static int e4m3_exact(float x) { if (x == 0) return 1; int e; const double m = frexp(fabs((double)x), &e) * 16; return m == floor(m) && e > -8 && e < 10; }
static void test_input(void) {
    uint8_t v[256]; for (int i = 0; i < 256; i++) v[i] = (uint8_t)i;
    uint8_t *dv = nn_malloc(256); void *dx = nn_malloc(256 * 2 * 2); nn_h2d(dv, v, 256);
    nn_restore_input(dv, dx, 256, 1, 1);
    _Float16 h[512]; nn_d2h(h, dx, sizeof h);
    int bad = 0, nex = 0;
    for (int i = 0; i < 256; i++) { const float hi = (float)h[i], lo = (float)h[256 + i]; bad += hi + lo / 4 != (i - 128) / 64.f; nex += !e4m3_exact(hi) || !e4m3_exact(lo); }
    CHECK(!bad, "split input: %d values with hi + lo / 4 != (x - 128) / 64", bad);
    CHECK(!nex, "split input: %d values not exact in e4m3", nex);
    nn_restore_input(dv, dx, 256, 0, 1); nn_d2h(h, dx, 512);
    bad = 0; for (int i = 0; i < 256; i++) bad += (float)h[i] != (i - 128) / 64.f;
    CHECK(!bad, "plain input: %d values", bad);
    nn_free(dv); nn_free(dx);
}

/* the loss in double from the same arrays */
static double loss_ref(const float *lg, const uint8_t *lo, const uint8_t *hi, int D, int H, int W, double eps, double mw, double ew, double *g) {
    const size_t n = (size_t)D * H * W; double cnt = 0, ch = 0, se = 0, pairs = 0, es = 0;
    for (size_t i = 0; i < n; i++) if (hi[i]) { const double e = lg[i] + (lo[i] - (double)hi[i]) / 64; cnt++; ch += sqrt(e * e + eps * eps) - eps; se += e * e; }
    const size_t str[3] = {(size_t)H * W, (size_t)W, 1}; const int len[3] = {D, H, W};
    for (size_t i = 0; i < n; i++) if (hi[i]) for (int a = 0; a < 3; a++) {
        if ((int)((i / str[a]) % len[a]) == len[a] - 1 || !hi[i + str[a]]) continue;
        const double d = (lg[i] + (lo[i] - (double)hi[i]) / 64) - (lg[i + str[a]] + (lo[i + str[a]] - (double)hi[i + str[a]]) / 64);
        pairs++; es += sqrt(d * d + eps * eps) - eps;
        if (g) { const double gd = d / sqrt(d * d + eps * eps); g[i] += ew * gd; g[i + str[a]] -= ew * gd; }
    }
    if (g) for (size_t i = 0; i < n; i++) { g[i] = pairs > 0 ? g[i] / pairs : 0; if (hi[i]) { const double e = lg[i] + (lo[i] - (double)hi[i]) / 64; g[i] += (e / sqrt(e * e + eps * eps) + mw * e) / cnt; } }
    return ch / cnt + 0.5 * mw * se / cnt + (pairs > 0 ? ew * es / pairs : 0);
}
static double gpu_loss(const float *dlg, const uint8_t *dlo, const uint8_t *dhi, int D, int H, int W, float eps, float mw, float ew, float *dg, double *dst) {
    nn_restore_loss(dlg, 0, dlo, dhi, D, H, W, eps, mw, ew, dg, 0, 1, 1.f, dst);
    double s[8]; nn_d2h(s, dst, sizeof s);
    return s[1] / s[0] + 0.5 * mw * s[3] / 4096 / s[0] + (s[5] > 0 ? ew * s[6] / s[5] : 0);
}
static void test_loss(void) {
    enum { D = 10, H = 12, W = 14 }; const size_t n = (size_t)D * H * W;
    float *lg = malloc(n * 4), *g = malloc(n * 4); uint8_t *lo = malloc(n), *hi = malloc(n); double *gr = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; i++) { hi[i] = urand() < 0.15 ? 0 : (uint8_t)(1 + 254 * urand()); lo[i] = (uint8_t)(1 + 254 * urand()); lg[i] = (float)(0.3 * (urand() - 0.5)); }
    const float eps = 0.01f, mw = 0.1f, ew = 0.5f;
    float *dlg = nn_malloc(n * 4), *dg = nn_malloc(n * 4); uint8_t *dlo = nn_malloc(n), *dhi = nn_malloc(n); double *dst = nn_malloc(8 * sizeof(double));
    nn_h2d(dlg, lg, n * 4); nn_h2d(dlo, lo, n); nn_h2d(dhi, hi, n);
    const double lr = loss_ref(lg, lo, hi, D, H, W, eps, mw, ew, gr), lgpu = gpu_loss(dlg, dlo, dhi, D, H, W, eps, mw, ew, dg, dst);
    CHECK(fabs(lr - lgpu) < 1e-5 * fabs(lr) + 1e-7, "loss %g vs double %g", lgpu, lr);
    nn_d2h(g, dg, n * 4);
    double mx = 0, gm = 0; for (size_t i = 0; i < n; i++) { mx = fmax(mx, fabs(g[i] - gr[i])); gm = fmax(gm, fabs(gr[i])); }
    CHECK(mx < 1e-4 * gm + 1e-9, "gradient vs double: max err %g (max |g| %g)", mx, gm);
    for (int t = 0; t < 12; t++) {   /* central differences of the GPU loss */
        size_t i = (size_t)(urand() * n); const float h = 2e-3f, l0 = lg[i];
        lg[i] = l0 + h; nn_h2d(dlg, lg, n * 4); const double lp = gpu_loss(dlg, dlo, dhi, D, H, W, eps, mw, ew, nullptr, dst);
        lg[i] = l0 - h; nn_h2d(dlg, lg, n * 4); const double lm = gpu_loss(dlg, dlo, dhi, D, H, W, eps, mw, ew, nullptr, dst);
        lg[i] = l0; nn_h2d(dlg, lg, n * 4);
        const double fd = (lp - lm) / (2 * h);
        CHECK(fabs(fd - g[i]) < 2e-3 * fmax(fabs(g[i]), 1e-3) + 2e-6, "central difference at %zu: %g vs gradient %g", i, fd, g[i]);
    }
    /* 16-bit gradient scaled by the gradient scale */
    nn_restore_loss(dlg, 0, dlo, dhi, D, H, W, eps, mw, ew, dg, 1, 1, 1024.f, dst);
    _Float16 *g16 = malloc(n * 2); nn_d2h(g16, dg, n * 2);
    mx = 0; for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs((double)g16[i] / 1024 - g[i]));
    CHECK(mx < 2e-3 * gm, "fp16 scaled gradient: max err %g", mx);
    nn_free(dlg); nn_free(dg); nn_free(dlo); nn_free(dhi); nn_free(dst); free(lg); free(g); free(lo); free(hi); free(gr); free(g16);
}

/* identity at step 0: a fresh net with the zero head outputs its input exactly */
static void test_identity(void) {
    unet_cfg cfg = {2, {8, 16}, 2, 1, 8, 0, 0};
    unet *u = unet_create(&cfg); unet_init(u, 5); unet_zero_head(u);
    enum { P = 32 }; const size_t n = (size_t)P * P * P;
    uint8_t *c = malloc(n), *o = malloc(n); ct_like(c, P, P, P, 1);
    uint8_t *dc = nn_malloc(n), *dout = nn_malloc(n); void *dx = nn_malloc(n * 4); nn_h2d(dc, c, n);
    nn_restore_input(dc, dx, n, 1, 1);
    const float *lg = unet_forward_x(u, dx, (shape5){1, 2, P, P, P}, 1, 1);
    nn_restore_output(lg, unet_logits_h16(u), dc, dout, n); nn_d2h(o, dout, n);
    CHECK(!memcmp(o, c, n), "zero-head output != input");
    unet_use_ema(u, 1); lg = unet_forward_x(u, dx, (shape5){1, 2, P, P, P}, 0, 1);
    nn_restore_output(lg, unet_logits_h16(u), dc, dout, n); nn_d2h(o, dout, n);
    CHECK(!memcmp(o, c, n), "zero-head EMA output != input");
    nn_free(dc); nn_free(dout); nn_free(dx); unet_free(u); free(c); free(o);
}

/* a local zarr (level "0") of a D^3 CT-like volume, written whole */
static void write_vol(const char *dir, const uint8_t *v, int N, int shard) {
    const int64_t sh[3] = {N, N, N}; z3w *w = z3w_create(dir, sh, shard, 0, 0, nullptr);
    uint8_t *b = malloc((size_t)shard * shard * shard);
    const int ns = (N + shard - 1) / shard, e = N < shard ? N : shard;
    memset(b, 0, (size_t)shard * shard * shard);
    for (int sz = 0; sz < ns; sz++) for (int sy = 0; sy < ns; sy++) for (int sx = 0; sx < ns; sx++) {
        for (int z = 0; z < e; z++) for (int y = 0; y < e; y++) memcpy(b + ((size_t)z * shard + y) * shard, v + ((size_t)(sz * shard + z) * N + sy * shard + y) * N + sx * shard, e);
        CHECK(!z3w_write_shard(w, sz, sy, sx, b, 4), "write shard");
    }
    CHECK(!z3w_close(w), "close"); free(b);
}
static void test_sampler(const char *tmp) {
    enum { N = 256, P = 64 }; const size_t n3 = (size_t)N * N * N, p3 = (size_t)P * P * P;
    uint8_t *v = malloc(n3); ct_like(v, N, N, N, 0);
    for (size_t i = 0; i < n3; i++) if (urand() < 0.05) v[i] = 0;
    char d[600], js[600]; snprintf(d, sizeof d, "%s/vol.zarr/0", tmp); write_vol(d, v, N, 128);
    snprintf(js, sizeof js, "%s/sources.json", tmp);
    FILE *f = fopen(js, "w");
    fprintf(f, "{\"sources\":[{\"name\":\"a\",\"root\":\"%s\",\"ct\":\"vol.zarr\",\"um\":2.4,\"boxes\":[[0,0,0,128,128,128],[128,128,128,128,128,128]],\"holdout\":[0,128,0,128,128,128]}]}\n", tmp);
    fclose(f);
    sources *S = sources_load(js); CHECK(S && S->n == 1 && S->src[0].nbox == 2, "sources with boxes");
    if (!S) return;
    for (int val = 0; val < 2; val++) for (int aug = 0; aug < 2; aug++) {
        if (val && aug) continue;
        sample_cfg c = sample_cfg_default(); c.P = P; c.B = 1; c.nworkers = 2; c.nbuf = 2; c.seed = 3; c.restore = 1; c.min_fg = 0.1; c.augment = aug; c.holdout = val; c.deterministic = 1;
        memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
        sampler *sp = sampler_start(S, &c); CHECK(sp != nullptr, "sampler_start restore");
        if (!sp) continue;
        for (int k = 0; k < 6; k++) {
            batch *b = sampler_next(sp); CHECK(b != nullptr, "batch"); if (!b) break;
            const int64_t *o = b->corner[0];
            int inside = 0;
            if (val) inside = o[0] >= 0 && o[0] + P <= 128 && o[1] >= 128 && o[1] + P <= 256 && o[2] >= 0 && o[2] + P <= 128;
            else inside = (o[0] + P <= 128 && o[1] + P <= 128 && o[2] + P <= 128) || (o[0] >= 128 && o[1] >= 128 && o[2] >= 128);
            CHECK(inside, "%s window %lld,%lld,%lld outside its box", val ? "validation" : "training", (long long)o[0], (long long)o[1], (long long)o[2]);
            CHECK(b->x == nullptr && b->x16 == nullptr && b->m == nullptr && b->w[0] == 1, "restore batch layout");
            size_t zw = 0, zt = 0, bad = 0;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
                const uint8_t s = v[((size_t)(o[0] + z) * N + o[1] + y) * N + o[2] + x], t = b->t[((size_t)z * P + y) * P + x];
                zw += !s; zt += !t; if (!aug) bad += s != t;
            }
            if (!aug) CHECK(!bad, "clean window != the CT at the corner (%zu voxels)", bad);
            else CHECK(zw == zt, "augmented window: %zu zeros vs %zu in the CT (LUT and symmetry keep 0)", zt, zw);
            (void)p3;
            sampler_release(sp, b);
        }
        sampler_stop(sp);
    }
    sources_free(S); free(v);
}

/* predict on a restore checkpoint: a zero-head net restores a coarse volume to its trilinear upsampling (codec off) */
static void test_predict(const char *tmp) {
    enum { N = 64, F = 4 };
    uint8_t *v = malloc((size_t)N * N * N); ct_like(v, N, N, N, 1);
    char d[600], ck[600], out[600]; snprintf(d, sizeof d, "%s/coarse.zarr/0", tmp); write_vol(d, v, N, 128);
    unet_cfg cfg = {2, {8, 16}, 2, 1, 8, 0, 0};
    unet *u = unet_create(&cfg); unet_init(u, 7); unet_zero_head(u);
    snprintf(ck, sizeof ck, "%s/r.ckpt", tmp);
    CHECK(!unet_save(u, ck, 0, "{\"task\":\"restore\",\"restore\":{\"version\":1,\"input\":\"split\",\"fine_um\":2.4}}"), "save");
    unet_free(u);
    snprintf(out, sizeof out, "%s/sr.zarr", tmp);
    char root[600]; snprintf(root, sizeof root, "%s", tmp);
    char *argv[] = {"ufsm", "predict", ck, root, "coarse.zarr", out, "--um", "9.6", "--shard", "128", "--halo", "16", "--q", "0", "--threads", "2", "--ema", "0", nullptr};
    int restore_predict(int argc, char **argv);
    CHECK(restore_predict(18, argv) == 0, "restore_predict");
    char lv[700]; snprintf(lv, sizeof lv, "%s/sr.zarr", tmp);
    store *s = store_open(lv); z3 *z = s ? z3_open(s, "2.4", nullptr) : nullptr;
    CHECK(z != nullptr, "open the output (level 2.4)");
    if (z) {
        const int M = N * F; const size_t m3 = (size_t)M * M * M;
        CHECK(z3_meta_of(z)->shape[0] == M && z3_meta_of(z)->shard[0] == 128, "output shape %lld shard %d", (long long)z3_meta_of(z)->shape[0], z3_meta_of(z)->shard[0]);
        uint8_t *o = malloc(m3), *ref = malloc(m3); const int64_t o0[3] = {0, 0, 0}, nn3[3] = {M, M, M};
        CHECK(!z3_read(z, o0, nn3, o, 4), "read the output");
        const float st[3] = {F, F, F}, ph[3] = {0, 0, 0};
        up_ref(v, N, N, N, ref, M, M, M, st, ph, nullptr);
        CHECK(near_equal(o, ref, m3, 1, 0.999, "predict"), "predict restore output != trilinear upsampling of the input");
        free(o); free(ref);
    }
    free(v);
}


/* ---- real pairs (src/srpair.h, nn_restore_pair_loss) ---- */
static double pair_ref(const float *lg, const uint8_t *lo, const uint8_t *hi, const uint8_t *co, int N, const restore_pair_w *w, double *g) {
    const int q = N / 4; const size_t n = (size_t)N * N * N; const double eps = w->eps;
    const size_t str[3] = {(size_t)N * N, (size_t)N, 1};
    double cnt = 0, main_ = 0, se = 0, pairs = 0, es = 0, dsn = 0, dss = 0;
    double *gm = calloc(n, sizeof *gm), *ge = calloc(n, sizeof *ge), *gd = calloc(n, sizeof *gd);
#define O(i) (lg[i] + lo[i] / 64.0)
#define CH(e) (sqrt((e) * (e) + eps * eps) - eps)
#define CD(e) ((e) / sqrt((e) * (e) + eps * eps))
    for (size_t i = 0; i < n; i++) if (hi[i]) {
        const double o = O(i), e = o - hi[i] / 64.0; double em = e;
        if (w->shift) for (int a = 0; a < 3; a++) { const int c = (int)((i / str[a]) % N);
            if (c > 0 && hi[i - str[a]]) { const double t = o - hi[i - str[a]] / 64.0; if (fabs(t) < fabs(em)) em = t; }
            if (c < N - 1 && hi[i + str[a]]) { const double t = o - hi[i + str[a]] / 64.0; if (fabs(t) < fabs(em)) em = t; } }
        cnt++; main_ += (1 - w->shift_w) * CH(e) + w->shift_w * CH(em); se += e * e;
        gm[i] = (1 - w->shift_w) * CD(e) + w->shift_w * CD(em) + w->mse * e;
        for (int a = 0; a < 3; a++) { if ((int)((i / str[a]) % N) == N - 1 || !hi[i + str[a]]) continue;
            const size_t j = i + str[a]; const double d = e - (O(j) - hi[j] / 64.0); pairs++; es += CH(d); ge[i] += CD(d); ge[j] -= CD(d); }
    }
    for (int cz = 0; cz < q; cz++) for (int cy = 0; cy < q; cy++) for (int cx = 0; cx < q; cx++) {
        double s = 0; int k = 0;
        for (int dz = 0; dz < 4; dz++) for (int dy = 0; dy < 4; dy++) for (int dx = 0; dx < 4; dx++) { const size_t i = ((size_t)(4 * cz + dz) * N + 4 * cy + dy) * N + 4 * cx + dx; if (lo[i]) { s += O(i); k++; } }
        const size_t c = ((size_t)cz * q + cy) * q + cx;
        if (k < 64 || !co[c]) continue;
        const double r = s / 64 - co[c] / 64.0; dss += CH(r); dsn++;
        for (int dz = 0; dz < 4; dz++) for (int dy = 0; dy < 4; dy++) for (int dx = 0; dx < 4; dx++) gd[((size_t)(4 * cz + dz) * N + 4 * cy + dy) * N + 4 * cx + dx] += CD(r) / 64;
    }
    if (g) for (size_t i = 0; i < n; i++) g[i] = gm[i] / cnt + (pairs > 0 ? w->edge * ge[i] / pairs : 0) + (dsn > 0 ? w->ds * gd[i] / dsn : 0);
    free(gm); free(ge); free(gd);
#undef O
#undef CH
#undef CD
    return main_ / cnt + 0.5 * w->mse * se / cnt + (pairs > 0 ? w->edge * es / pairs : 0) + (dsn > 0 ? w->ds * dss / dsn : 0);
}
static double pair_gpu(const float *dlg, const uint8_t *dlo, const uint8_t *dhi, const uint8_t *dco, int N, const restore_pair_w *w, float *dg, double *dst, float *drd, double *s) {
    nn_restore_pair_loss(dlg, 0, dlo, dhi, dco, N, N, N, w, dg, 0, 1, 1.f, dst, drd, 1);
    nn_d2h(s, dst, RP_NST * sizeof(double));
    return s[1] / s[0] + 0.5 * w->mse * s[3] / 4096 / s[0] + (s[5] > 0 ? w->edge * s[6] / s[5] : 0) + (s[8] > 0 ? w->ds * s[7] / s[8] : 0);
}
static void test_pair_loss(void) {
    enum { N = 16 }; const size_t n = (size_t)N * N * N, nq = n / 64;
    float *lg = malloc(n * 4), *g = malloc(n * 4); uint8_t *lo = malloc(n), *hi = malloc(n), *co = malloc(nq); double *gr = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; i++) { hi[i] = urand() < 0.1 ? 0 : (uint8_t)(1 + 254 * urand()); lo[i] = urand() < 0.02 ? 0 : (uint8_t)(1 + 254 * urand()); lg[i] = (float)(0.3 * (urand() - 0.5)); }
    for (size_t c = 0; c < nq; c++) co[c] = (uint8_t)(1 + 254 * urand());
    float *dlg = nn_malloc(n * 4), *dg = nn_malloc(n * 4), *drd = nn_malloc(nq * 4); uint8_t *dlo = nn_malloc(n), *dhi = nn_malloc(n), *dco = nn_malloc(nq); double *dst = nn_malloc(RP_NST * sizeof(double)), s[RP_NST];
    nn_h2d(dlg, lg, n * 4); nn_h2d(dlo, lo, n); nn_h2d(dhi, hi, n); nn_h2d(dco, co, nq);
    for (int shift = 0; shift < 2; shift++) {
        const restore_pair_w w = {0.01f, 0.1f, 0.3f, shift ? 0.5f : 0.f, 0.7f, shift, 0.f};
        const double lr = pair_ref(lg, lo, hi, co, N, &w, gr), lgpu = pair_gpu(dlg, dlo, dhi, dco, N, &w, dg, dst, drd, s);
        CHECK(fabs(lr - lgpu) < 1e-5 * fabs(lr) + 1e-7, "pair loss (shift %d) %g vs double %g", shift, lgpu, lr);
        nn_d2h(g, dg, n * 4);
        double mx = 0, gm = 0; for (size_t i = 0; i < n; i++) { mx = fmax(mx, fabs(g[i] - gr[i])); gm = fmax(gm, fabs(gr[i])); }
        CHECK(mx < 1e-4 * gm + 1e-9, "pair gradient (shift %d) vs double: max err %g (max |g| %g)", shift, mx, gm);
        CHECK(s[8] > 0 && s[13] == 0, "pair stats: ds count %g, ssim blocks %g (every 8^3 block holds an invalid voxel)", s[8], s[13]);
        if (!shift) for (int t = 0; t < 12; t++) {   /* central differences (the shift-min term switches branches: shift 0 only) */
            size_t i = (size_t)(urand() * n); const float h = 2e-3f, l0 = lg[i];
            lg[i] = l0 + h; nn_h2d(dlg, lg, n * 4); const double lp = pair_gpu(dlg, dlo, dhi, dco, N, &w, nullptr, dst, drd, s);
            lg[i] = l0 - h; nn_h2d(dlg, lg, n * 4); const double lm = pair_gpu(dlg, dlo, dhi, dco, N, &w, nullptr, dst, drd, s);
            lg[i] = l0; nn_h2d(dlg, lg, n * 4);
            const double fd = (lp - lm) / (2 * h);
            CHECK(fabs(fd - g[i]) < 2e-3 * fmax(fabs(g[i]), 1e-3) + 2e-6, "pair central difference at %zu: %g vs gradient %g", i, fd, g[i]);
        }
    }
    {   /* the scale head (--sr-unc): channel 1 = log b; its gradient matches central differences of unc * NLL / count, channel 0 unchanged */
        const restore_pair_w wu = {0.01f, 0.1f, 0.3f, 0.f, 0.7f, 0, 0.5f};
        float *l2 = malloc(2 * n * 4), *g2 = malloc(2 * n * 4), *dl2 = nn_malloc(2 * n * 4), *dg2 = nn_malloc(2 * n * 4);
        memcpy(l2, lg, n * 4); for (size_t i = 0; i < n; i++) l2[n + i] = (float)(urand() * 2 - 1);
        nn_h2d(dl2, l2, 2 * n * 4);
#define PU(gp) (pair_gpu(dl2, dlo, dhi, dco, N, &wu, gp, dst, drd, s) + wu.unc * s[15] / s[0])
        PU(dg2); nn_d2h(g2, dg2, 2 * n * 4);
        const restore_pair_w w0 = {0.01f, 0.1f, 0.3f, 0.f, 0.7f, 0, 0.f};
        pair_gpu(dlg, dlo, dhi, dco, N, &w0, dg, dst, drd, s); nn_d2h(g, dg, n * 4);
        double mx = 0; for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs(g2[i] - g[i]));
        CHECK(mx < 1e-7, "scale head changes the channel-0 gradient (max %g)", mx);
        for (int t = 0; t < 10; t++) {
            size_t i = n + (size_t)(urand() * n); const float h = 2e-3f, l0 = l2[i];
            l2[i] = l0 + h; nn_h2d(dl2, l2, 2 * n * 4); const double lp = PU(nullptr);
            l2[i] = l0 - h; nn_h2d(dl2, l2, 2 * n * 4); const double lm = PU(nullptr);
            l2[i] = l0; nn_h2d(dl2, l2, 2 * n * 4);
            const double fd = (lp - lm) / (2 * h);
            CHECK(fabs(fd - g2[i]) < 2e-3 * fmax(fabs(g2[i]), 1e-3) + 2e-6, "scale head central difference at %zu: %g vs gradient %g", i, fd, g2[i]);
        }
#undef PU
        free(l2); free(g2); nn_free(dl2); nn_free(dg2);
    }
    /* SSIM of a perfect output is 1 */
    for (size_t i = 0; i < n; i++) { hi[i] = (uint8_t)(1 + 254 * urand()); lo[i] = (uint8_t)(1 + 254 * urand()); lg[i] = ((float)hi[i] - lo[i]) / 64.f; }
    nn_h2d(dlg, lg, n * 4); nn_h2d(dlo, lo, n); nn_h2d(dhi, hi, n);
    const restore_pair_w w = {0.01f, 0.1f, 0.3f, 0.5f, 0.7f, 1, 0.f};
    pair_gpu(dlg, dlo, dhi, dco, N, &w, nullptr, dst, drd, s);
    CHECK(s[13] == 8 && fabs(s[11] / s[13] - 1) < 1e-4 && s[3] < 1e-6 && s[9] < 1e-6, "perfect output: ssim %g over %g blocks, sse %g, grad sse %g", s[11] / s[13], s[13], s[3], s[9]);
    nn_free(dlg); nn_free(dg); nn_free(drd); nn_free(dlo); nn_free(dhi); nn_free(dco); nn_free(dst); free(lg); free(g); free(lo); free(hi); free(co); free(gr);
}
/* a z-stacked array (nz, e, e), e <= shard, written as shards along z */
static void write_stack(const char *dir, const uint8_t *v, int nz, int e, int shard) {
    const int64_t sh[3] = {nz, e, e}; z3w *w = z3w_create(dir, sh, shard, 0, 0, nullptr);
    uint8_t *b = calloc((size_t)shard * shard * shard, 1);
    for (int sz = 0; sz * shard < nz; sz++) {
        memset(b, 0, (size_t)shard * shard * shard);
        for (int z = 0; z < shard && sz * shard + z < nz; z++) for (int y = 0; y < e; y++) memcpy(b + ((size_t)z * shard + y) * shard, v + ((size_t)(sz * shard + z) * e + y) * e, e);
        CHECK(!z3w_write_shard(w, sz, 0, 0, b, 4), "write shard");
    }
    CHECK(!z3w_close(w), "close"); free(b);
}
static void test_pair_sampler(const char *tmp) {
    enum { B = 16, T = 64, NB = 3, P = 32 }; const size_t b3 = (size_t)B * B * B, t3 = (size_t)T * T * T, p3 = (size_t)P * P * P;
    uint8_t *in = malloc(NB * b3), *tg = malloc(NB * t3);
    for (size_t i = 0; i < NB * b3; i++) in[i] = (uint8_t)(1 + 254 * urand());
    for (size_t i = 0; i < NB * t3; i++) tg[i] = urand() < 0.02 ? 0 : (uint8_t)(1 + 254 * urand());
    char d[600], p[700]; snprintf(d, sizeof d, "%s/pair", tmp);
    snprintf(p, sizeof p, "%s/input.zarr", d); write_stack(p, in, NB * B, B, 128);
    snprintf(p, sizeof p, "%s/target.zarr", d); write_stack(p, tg, NB * T, T, 128);
    snprintf(p, sizeof p, "%s/boxes.json", d); FILE *f = fopen(p, "w");
    fprintf(f, "{\"input\":{\"um\":9.36,\"box\":%d},\"target\":{\"key\":\"x/y-2.4um-0.2m-77keV.zarr\",\"box\":%d},\"boxes\":[{\"slot\":0,\"split\":\"train\"},{\"slot\":1,\"split\":\"test\"},{\"slot\":2,\"split\":\"train\"}]}\n", B, T); fclose(f);
    snprintf(p, sizeof p, "%s/intensity_map.json", d); f = fopen(p, "w"); fprintf(f, "{\"lut\":[");
    for (int v = 0; v < 256; v++) fprintf(f, "%s%.3f", v ? "," : "", 0.5 * v + 20); fprintf(f, "]}\n"); fclose(f);
    char js[600]; snprintf(js, sizeof js, "%s/pairs.json", tmp); f = fopen(js, "w");
    fprintf(f, "{\"sources\":[{\"name\":\"pr\",\"root\":\"%s\",\"ct\":\"input.zarr\",\"um\":9.36,\"sr_pair\":{\"role\":\"train\"}},{\"name\":\"prc\",\"root\":\"%s\",\"ct\":\"input.zarr\",\"um\":9.36,\"weight\":1,\"sr_pair\":{\"role\":\"train\",\"cache\":1}}]}\n", d, d); fclose(f);
    sources *S = sources_load(js);
    CHECK(S && S->n == 2 && S->src[1].srp && S->src[1].srp->cache && S->src[0].srp && S->src[0].srp->ntr == 2 && S->src[0].srp->nte == 1 && S->src[0].srp->keV == 77 && S->src[0].srp->lut[10] == 25 && S->src[0].srp->lut[0] == 0, "pair source");
    if (!S || !S->src[0].srp) return;
    for (int val = 0; val < 2; val++) {
        sample_cfg c = sample_cfg_default(); c.P = P; c.B = 1; c.nworkers = 2; c.nbuf = 2; c.seed = 5; c.restore = 2; c.min_fg = 0.5; c.augment = 0; c.holdout = val; c.deterministic = 1;
        memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
        sampler *sp = sampler_start(S, &c); CHECK(sp != nullptr, "sampler_start pairs");
        if (!sp) continue;
        for (int k = 0; k < 6; k++) {
            batch *b = sampler_next(sp); CHECK(b != nullptr, "pair batch"); if (!b) break;
            const int64_t *o = b->corner[0]; const int box = (int)(o[0] / B), oc[3] = {(int)(o[0] % B), (int)o[1], (int)o[2]};
            CHECK(val ? box == 1 : box != 1, "%s window from box %d", val ? "validation" : "training", box);
            CHECK(fabs(b->um[0] - 9.36 / 4) < 1e-5, "pair batch um %g", b->um[0]);
            size_t bad = 0;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
                const uint8_t t = tg[(size_t)box * t3 + ((size_t)(4 * oc[0] + z) * T + 4 * oc[1] + y) * T + 4 * oc[2] + x];
                bad += b->t[((size_t)z * P + y) * P + x] != (t ? S->src[b->src[0]].srp->lut[t] : 0);
            }
            for (int z = 0; z < P / 4; z++) for (int y = 0; y < P / 4; y++) for (int x = 0; x < P / 4; x++)
                bad += b->t[p3 + ((size_t)z * (P / 4) + y) * (P / 4) + x] != in[(size_t)box * b3 + ((size_t)(oc[0] + z) * B + oc[1] + y) * B + oc[2] + x];
            CHECK(!bad, "pair batch: %zu voxels differ from the target (through the LUT) / input crop", bad);
            sampler_release(sp, b);
        }
        sampler_stop(sp);
    }
    sources_free(S); free(in); free(tg);
}

int main(void) {
    if (nn_init(0)) { fprintf(stderr, "no GPU\n"); return 1; }
    nn_set_f16(1); nn_set_grad_scale(1024.f);
    char tmp[] = "/tmp/ufsm_restore_XXXXXX"; if (!mkdtemp(tmp)) return 1;
    test_codec(); test_degrade(); test_input(); test_loss(); test_identity(); test_sampler(tmp); test_predict(tmp); test_pair_loss(); test_pair_sampler(tmp);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", tmp); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_restore_task: %d failures\n", fails); return 1; }
    printf("test_restore_task: ok\n");
    return 0;
}
