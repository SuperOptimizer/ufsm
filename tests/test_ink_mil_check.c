/* train --ink-mil checks beside test_ink_mil_check (the map against the image warp, the loss sum): the device MIL path against the host
   reference, determinism, and the rays against the ink target plane the loss reads, through every geometry path. On the synthetic spiral
   (tests/make_wind_fixture.c) with one tifxyz segment on a papyrus sheet (blob labels) as the source's onfly "ink" target, 6-step runs:
   plain twice under --det 1, warped (rotation + elastic) with host label warps, warped with device label warps (--gpu-aug 2), the latent
   head stream (--lat 4 --lat-guide 3), and --replay 3 (replays re-draw the symmetry / warp on the device: the rays re-mapped from the
   source grid, sample.c mil0). UFSM_MIL_CHECK=1 in head_slab_pass:
   (1) every window's device MIL loss (fwd_slab per slab, fin) equals ink_mil_loss_host on the same fp16 logits (relative 1e-4), and the
       gradient pass 1 adds (bwd_slab: after - before, 16-bit) equals the host gradient within one 16-bit ulp; the host reference is
       central-difference tested in test_ink_mil, so this is the central-difference check through head_slab_pass at the logits;
   (2) the rays meet the window's known ink (the plane head_slab_pass reads) within +-R, with the ray's label, on >= 90% of rays in every
       run (the replay run alone: the remap);
   (3) --det 1 twice: the logged loss terms (ink mil included) are identical; the ink mil term is logged, finite, on windows with rays.
   Usage: test_ink_mil_check <ufsm binary> <make_wind_fixture binary> */
#define WIND_FIXTURE_NO_MAIN
#include "../tests/make_wind_fixture.c"
#include "tiffw.h"
#include <stdarg.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int sh(const char *fmt, ...) {
    char cmd[8192]; va_list a; va_start(a, fmt); vsnprintf(cmd, sizeof cmd, fmt, a); va_end(a);
    const int r = system(cmd);
    if (r) fprintf(stderr, "command failed (%d): %s\n", r, cmd);
    return r;
}
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1); if (!s) { fclose(f); return nullptr; }
    s[fread(s, 1, (size_t)n, f)] = 0; fclose(f); return s;
}
/* the step lines from " loss " through the gradient norm (as test_heads compares) */
static char *loss_terms(const char *log) {
    size_t cap = 1 << 16, n = 0; char *o = malloc(cap); o[0] = 0;
    for (const char *l = log; l && *l; ) {
        const char *e = strchr(l, '\n'); const size_t len = e ? (size_t)(e - l) : strlen(l);
        if (len > 5 && !strncmp(l, "step ", 5)) {
            const char *a = strstr(l, " loss "); const char *b = a ? strstr(a, " gn ") : nullptr;   /* (through the gradient norm: the timing follows) */
            if (b) { b += 4; while (*b && *b != ' ' && *b != '\n') b++; }
            if (a && a < l + len) { const char *t = b && b < l + len ? b : l + len; size_t k = (size_t)(t - a);
                if (n + k + 2 < cap) { memcpy(o + n, a, k); n += k; o[n++] = '\n'; o[n] = 0; } }
        }
        l = e ? e + 1 : nullptr;
    }
    return o;
}
enum { N = 256 };
static const double W0 = 3.15, STEP = 4.0;   /* the segment: winding 3.15 (mid papyrus: the fixture's sheets are f < 0.3), 4 voxels a grid step */
static double r_at(double th, double z) { return 24.0 * (W0 + th / (2 * M_PI) - .002 * z); }   /* fx_wind = W0 */

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_ink_mil_check <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    char d[] = "/tmp/test_inkmilc_XXXXXX"; if (!mkdtemp(d)) return 1;
    CHECK(!sh("%s %s/fx %d 2 0 > %s/fx.log 2>&1", argv[2], d, N, d), "fixture");
    fx_N = N;
    /* the segment: angle th in [-1.2, 1.2] rad (columns, ~STEP voxels apart at r 75), z in [40, 216] (rows) */
    const double th0 = -1.2, th1 = 1.2, za = 40, zb = 216, dth = STEP / 75.0;
    const int GW = (int)((th1 - th0) / dth) + 1, GH = (int)((zb - za) / STEP) + 1;
    float *X = malloc((size_t)GW * GH * 4), *Y = malloc((size_t)GW * GH * 4), *Z = malloc((size_t)GW * GH * 4);
    for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) {
        const double z = za + STEP * v, th = th0 + dth * u, r = r_at(th, z); double cy, cx; fx_axis_at(z, &cy, &cx);
        X[v * GW + u] = (float)(cx + r * cos(th)); Y[v * GW + u] = (float)(cy + r * sin(th)); Z[v * GW + u] = (float)z;
    }
    char p[1400], seg[1100], lab[1100], msk[1100];
    snprintf(seg, sizeof seg, "%s/seg.tifxyz", d);
    CHECK(!tifxyz_write(seg, GW, GH, X, Y, Z, STEP, 9.6, -1, "test"), "tifxyz");
    const int CW = (int)(GW * STEP), CH = (int)(GH * STEP);   /* the canvas: ink in 30-pixel discs on a 90-pixel lattice, all supervised */
    uint8_t *L = malloc((size_t)CW * CH), *M = malloc((size_t)CW * CH);
    for (int r = 0; r < CH; r++) for (int c = 0; c < CW; c++) { const int dr = r % 90 - 45, dc = c % 90 - 45; L[r * CW + c] = dr * dr + dc * dc < 30 * 30 ? 255 : 0; M[r * CW + c] = 255; }
    snprintf(lab, sizeof lab, "%s/inklabels.tif", d); snprintf(msk, sizeof msk, "%s/mask.tif", d);
    CHECK(!tiffw_write(lab, CW, CH, 1, L, 1) && !tiffw_write(msk, CW, CH, 1, M, 1), "label tifs");
    snprintf(p, sizeof p, "%s/segs.json", d); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"segments\": [{\"tifxyz\": \"%s\", \"labels\": \"%s\", \"mask\": \"%s\"}]}\n", seg, lab, msk); fclose(f);
    snprintf(p, sizeof p, "%s/sources-mil.json", d); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture-mil\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ink_weight\": 1,\n"
               "  \"targets\": {\"ink\": {\"onfly\": {\"segments\": \"%s/segs.json\", \"level\": 0, \"iso\": 100, \"cache_dir\": \"off\"}},\n"
               "               \"band\": {\"root\": \"%s/fx/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/fx/phase0\", \"key\": \"0\", \"level\": 0}}}}]}\n",
            d, d, d, d, d);
    fclose(f);
    const char *common = "--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 "
                         "--fp4 2 --input-prec 8 --warmup 0 --seed 7 --wind4-tile 0 --levels 1 --widths 16,32,64 --ink-head 1 --ink-smooth 0.1 --det 1";
    const char *mil = "--ink-mil 1 --ink-mil-r 3 --ink-mil-mode 0 --ink-mil-tau 0.5 --ink-mil-stride 1";
    for (int run = 0; run < 2; run++)
        CHECK(!sh("UFSM_MIL_CHECK=1 UFSM_GRAD_MX8=1 %s train %s/sources-mil.json --out %s/r%d --steps 6 %s %s > %s/r%d.log 2>&1", U, d, d, run, common, mil, d, run), "train run %d (%s/r%d.log)", run, d, run);
    /* run 2: warped windows (rotation + elastic, host label warps: the ray check reads the warped plane), run 3 the same warped on the device */
    const char *warp = "--geometry-aug 1 --rotate-p 1 --elastic-p 1";
    for (int run = 2; run < 4; run++)
        CHECK(!sh("UFSM_MIL_CHECK=1 UFSM_GRAD_MX8=1 %s train %s/sources-mil.json --out %s/r%d --steps 6 %s %s %s --gpu-aug %d > %s/r%d.log 2>&1", U, d, d, run, common, mil, warp, run == 2 ? 0 : 2, d, run), "train run %d (%s/r%d.log)", run, d, run);
    /* run 4: the latent net (u1's head stream, --lat 4 --lat-guide 3): the MIL gradient goes through the slab adjoint */
    CHECK(!sh("UFSM_MIL_CHECK=1 UFSM_GRAD_MX8=1 %s train %s/sources-mil.json --out %s/r4 --steps 6 %s %s --lat 4 --lat-guide 3 > %s/r4.log 2>&1", U, d, d, common, mil, d), "train run 4, latent (%s/r4.log)", d);
    /* run 5: --replay 3 with the warps (each window 3 steps, the replays with a new device-side symmetry / warp) */
    CHECK(!sh("UFSM_MIL_CHECK=1 UFSM_GRAD_MX8=1 %s train %s/sources-mil.json --out %s/r5 --steps 9 %s %s %s --gpu-aug 2 --replay 3 > %s/r5.log 2>&1", U, d, d, common, mil, warp, d), "train run 5, replays (%s/r5.log)", d);
    enum { NRUN = 6 };
    char *lg[NRUN]; for (int run = 0; run < NRUN; run++) { snprintf(p, sizeof p, "%s/r%d.log", d, run); lg[run] = slurp(p); if (!lg[run]) { fprintf(stderr, "FAIL: no log r%d in %s\n", run, d); return 1; } }
    /* (1) the device MIL loss / gradient against the host reference */
    int nl = 0, ng = 0, withrays = 0; double worst_l = 0, worst_g = 0;
    for (int run = 0; run < NRUN; run++) if (run != 1) for (const char *q = lg[run]; (q = strstr(q, "mil check: loss device ")); q++) {
        double ld, lh, rel; int nr;
        if (sscanf(q, "mil check: loss device %lf host %lf (relative %lf, %d rays)", &ld, &lh, &rel, &nr) == 4) { nl++; withrays += nr > 0; if (rel > worst_l) worst_l = rel; }
    }
    for (int run = 0; run < NRUN; run++) if (run != 1) for (const char *q = lg[run]; (q = strstr(q, "mil check: gradient rms difference")); q++) {
        double rms, mx, mg, ulp; int nr;
        if (sscanf(q, "mil check: gradient rms difference / rms %lf, max |difference| %lf of max |gradient| %lf, beyond one 16-bit ulp %lf (%d rays)", &rms, &mx, &mg, &ulp, &nr) == 5) {
            ng++; const double e = mg > 0 ? ulp / mg : ulp; if (e > worst_g) worst_g = e;
            CHECK(mg > 0, "a zero MIL gradient (%d rays)", nr);
        }
    }
    fprintf(stderr, "mil check: %d loss lines (worst relative %.3g), %d gradient lines (worst excess over one ulp / max %.3g)\n", nl, worst_l, ng, worst_g);
    CHECK(nl >= 10 && ng >= 10, "too few windows with rays: %d loss / %d gradient checks (%s/r0.log)", nl, ng, d);
    CHECK(worst_l < 1e-4, "device loss vs host: relative %.3g", worst_l);
    CHECK(worst_g < 1e-3, "device gradient vs host: %.3g beyond one 16-bit ulp (of the max)", worst_g);
    CHECK(strstr(lg[4], "mil check: gradient") != nullptr && strstr(lg[5], "mil check: gradient") != nullptr, "the latent / replay runs checked no MIL gradient (%s)", d);
    /* (2) the rays against the ink target plane, per run */
    for (int run = 0; run < NRUN; run++) {
        long rn = 0, rh = 0, ra = 0; int nw = 0;
        for (const char *q = lg[run]; (q = strstr(q, "mil check: rays ")); q++) { int n, h, a; if (sscanf(q, "mil check: rays %d, %d meet known ink within the reach, %d of those", &n, &h, &a) == 3) { rn += n; rh += h; ra += a; nw++; } }
        fprintf(stderr, "run %d: %d windows, %ld of %ld rays meet known ink, %ld of those agree with the label\n", run, nw, rh, rn, ra);
        CHECK(nw >= 1 && rh >= 0.9 * rn && ra >= 0.9 * rh, "run %d: the rays miss the window's ink plane: %ld of %ld meet it, %ld agree", run, rh, rn, ra);
    }
    /* (3) --det 1: the same loss terms */
    char *t0 = loss_terms(lg[0]), *t1 = loss_terms(lg[1]);
    CHECK(*t0 && !strcmp(t0, t1), "--det 1: the loss terms differ between runs:\n%s---\n%s", t0, t1);
    /* (3b) the ink mil term */
    const char *m = nullptr; double v = -1; int pct = -1, nterm = 0;   /* (the step lines of windows with rays) */
    for (const char *q = lg[0]; (q = strstr(q, " ink mil ")); q++) {
        double vv; int pp; if (sscanf(q, " ink mil %lf (%d%%", &vv, &pp) != 2) continue;
        nterm++; if (pp > 0 && !m) { m = q; v = vv; pct = pp; }
        CHECK(isfinite(vv) && (pp == 0 || vv > 0), "ink mil term %g (%d%% of windows)", vv, pp);
    }
    CHECK(nterm >= 6 && m && isfinite(v) && v > 0 && pct > 0, "the ink mil term (%d step lines, %s/r0.log)", nterm, d);
    if (m) { const char *e = strchr(m, '\n'); fprintf(stderr, "logged:%.*s\n", e ? (int)(e - m) : 80, m); }
    free(t0); free(t1); for (int run = 0; run < NRUN; run++) free(lg[run]);
    if (!fails) sh("rm -rf %s", d);
    fprintf(stderr, fails ? "test_ink_mil_check: %d FAILED (kept %s)\n" : "test_ink_mil_check: all passed\n", fails, d);
    return fails != 0;
}
