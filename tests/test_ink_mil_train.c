/* train --ink-mil: the normal-pooled MIL ink loss in the trainer (inkmil.h kernels in head_slab_pass, rays from the sampler).
   (1) sample_mil_map (CPU): a source-grid point / normal through the window's symmetry and warp lands where sample_warp_ink moves a
       blob centred there (the centroid of the warped blob within 0.15 voxels, the normal within 3 degrees of the warped finite step),
       for 8 symmetries x {identity, rotation + elastic, rotation + elastic + affine + zoom};
   (2) on the synthetic spiral (tests/make_wind_fixture.c, N 256) with one spiral sheet as a tifxyz segment (ink stripes on its
       canvas) and the source's "ink" target on the fly: train --task unwrap2 --ink-head 1 --ink-mil 1 --ink-dice 0.25 runs a few
       steps with the symmetries and warps on; the log's "ink mil" is finite, positive, on > 0 windows with rays; at --ink-mil 0 the
       first step's loss equals the --ink-mil run's minus its mil term (same seed, lr 0);
   (3) ufsm ink-rayface: the precomputed face-drop bits give MIL mode 1's rays byte for byte, and a mode-1 run on them logs the same
       first-step mil loss as one with the per-window face search (UFSM_INK_RAYFACE=0).
   Usage: test_ink_mil_train <ufsm binary> <make_wind_fixture binary> */
#include "inksplat.h"
#include "sample.h"
#include "spatial_augment.h"
#include "tiffw.h"
#include "imgops.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int sh(const char *fmt, ...) {
    char cmd[8192]; va_list a; va_start(a, fmt); vsnprintf(cmd, sizeof cmd, fmt, a); va_end(a);
    const int r = system(cmd); if (r) fprintf(stderr, "command failed (%d): %s\n", r, cmd); return r;
}

/* (1) the map against the image warp */
static void blob_centroid(const uint8_t *img, int P, double c[3]) {
    double s = 0, m[3] = {0};
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const int v = img[((size_t)z * P + y) * P + x]; if (v == 255 || !v) continue;
        s += v; m[0] += v * (double)z; m[1] += v * (double)y; m[2] += v * (double)x; }
    for (int d = 0; d < 3; d++) c[d] = s > 0 ? m[d] / s : -1;
}
static void blob(uint8_t *img, int P, const double p[3]) {   /* a smooth bump of radius 3 at p (0 elsewhere: known) */
    memset(img, 0, (size_t)P * P * P);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const double r2 = (z - p[0]) * (z - p[0]) + (y - p[1]) * (y - p[1]) + (x - p[2]) * (x - p[2]);
        if (r2 < 16) img[((size_t)z * P + y) * P + x] = (uint8_t)lrint(200 * (1 - r2 / 16) * (1 - r2 / 16)); }
}
static void test_map(void) {
    enum { P = 48 };
    uint8_t *a = malloc((size_t)P * P * P), *b = malloc((size_t)P * P * P);
    static const int perms[8][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2}, {2, 0, 1}};
    static const int flips[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 1}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}};
    double worst = 0, worst_n = 0; int n_ok = 0;
    for (int g = 0; g < 3; g++) for (int s = 0; s < 8; s++) {
        spatial_aug sp; spatial_aug_make(&sp, 1000 + 7 * s + g, P, g ? 12.f : 0.f, g ? 1.f : 0.f, g ? 1.5f : 0.f, g ? 1.f : 0.f);
        if (g == 2) { spatial_aug_affine(&sp, 99 + s, .08, .05); spatial_aug_zoom(&sp, .9); }
        sp.active = 1;   /* (sample_warp_ink always pulls through the map; the identity map is exact) */
        const double p[3] = {20.3 + s * .4, 25.7 - s * .3, 22.1 + .2 * s}, nv[3] = {.6, -.48, .64};
        inkray r; memset(&r, 0, sizeof r);
        for (int d = 0; d < 3; d++) { r.p[d] = (float)p[d]; r.n[d] = (float)nv[d]; }
        int kept = sample_mil_map(&r, 1, perms[s], flips[s], &sp, P, P);
        CHECK(kept == 1, "geometry %d symmetry %d: the ray left the window", g, s); if (kept != 1) continue;
        double c0[3], c1[3];
        blob(a, P, p); sample_warp_ink(a, b, P, perms[s], flips[s], &sp); blob_centroid(b, P, c0);
        const double p2[3] = {p[0] + 3 * nv[0], p[1] + 3 * nv[1], p[2] + 3 * nv[2]};   /* the normal: the warped blob of p + 3 n */
        blob(a, P, p2); sample_warp_ink(a, b, P, perms[s], flips[s], &sp); blob_centroid(b, P, c1);
        double e = 0, dn = 0, dl = 0;
        for (int d = 0; d < 3; d++) { e += (c0[d] - r.p[d]) * (c0[d] - r.p[d]); dl += (c1[d] - c0[d]) * (c1[d] - c0[d]); }
        dl = sqrt(dl); for (int d = 0; d < 3; d++) dn += (c1[d] - c0[d]) / dl * r.n[d];
        const double ang = acos(fmin(1, dn)) * 180 / M_PI; e = sqrt(e);
        if (e > worst) worst = e;
        if (ang > worst_n) worst_n = ang;
        n_ok += e < .15 && ang < 3;
        CHECK(e < .15 && ang < 3, "geometry %d symmetry %d: the mapped ray (%.2f %.2f %.2f) vs the warped blob (%.2f %.2f %.2f): %.3f voxels, normal %.2f deg",
              g, s, r.p[0], r.p[1], r.p[2], c0[0], c0[1], c0[2], e, ang);
    }
    fprintf(stderr, "sample_mil_map: %d of 24 maps within 0.15 voxels / 3 deg (worst %.3f voxels, %.2f deg)\n", n_ok, worst, worst_n);
    free(a); free(b);
}

/* (2) the fixture's spiral sheet (winding fraction 0.15, inside the papyrus band f < .3) as a tifxyz segment */
static void fx_axis_at(double z, double *cy, double *cx, int N) { *cy = .5 * N + .03 * (z - .5 * N); *cx = .5 * N - .02 * (z - .5 * N); }
static int write_segment(const char *dir, int N) {
    const double STEP = 4, f0 = .15, t0 = 1.2, t1 = 3.8;   /* turns along the sheet (radius 24 t .. at pitch 24) */
    const int GH = (int)((N - 16) / STEP), GW = (int)((t1 - t0) * 2 * M_PI * 60 / STEP);
    float *X = malloc((size_t)GW * GH * 4), *Y = malloc((size_t)GW * GH * 4), *Z = malloc((size_t)GW * GH * 4);
    for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) {
        const double z = 8 + STEP * v, ph = 2 * M_PI * (t0 + (t1 - t0) * u / (GW - 1)), r = 24 * (f0 + ph / (2 * M_PI) - .002 * z);
        double cy, cx; fx_axis_at(z, &cy, &cx, N);
        X[v * GW + u] = (float)(cx + r * cos(ph)); Y[v * GW + u] = (float)(cy + r * sin(ph)); Z[v * GW + u] = (float)z; }
    char p[600]; snprintf(p, sizeof p, "%s/seg.tifxyz", dir);
    if (tifxyz_write(p, GW, GH, X, Y, Z, STEP, 9.6, -1, "test_ink_mil_train")) return -1;
    const int CW = (int)(GW * STEP), CH = (int)(GH * STEP);
    uint8_t *lab = malloc((size_t)CW * CH), *msk = malloc((size_t)CW * CH);
    for (int r = 0; r < CH; r++) for (int c = 0; c < CW; c++) { lab[(size_t)r * CW + c] = ((r / 24 + c / 40) & 1) ? 255 : 0; msk[(size_t)r * CW + c] = 255; }
    snprintf(p, sizeof p, "%s/inklabels.tif", dir); if (tiffw_write(p, CW, CH, 1, lab, 1)) return -1;
    snprintf(p, sizeof p, "%s/mask.tif", dir); if (tiffw_write(p, CW, CH, 1, msk, 1)) return -1;
    snprintf(p, sizeof p, "%s/segs.json", dir); FILE *f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "{\"segments\": [{\"tifxyz\": \"%s/seg.tifxyz\", \"labels\": \"%s/inklabels.tif\", \"mask\": \"%s/mask.tif\"}]}\n", dir, dir, dir); fclose(f);
    free(X); free(Y); free(Z); free(lab); free(msk); return 0;
}
static double log_last(const char *path, const char *key, int *found) {   /* the value after the last "key" in the log */
    FILE *f = fopen(path, "r"); *found = 0; if (!f) return NAN;
    char ln[8192]; double v = NAN;
    while (fgets(ln, sizeof ln, f)) { const char *q = strstr(ln, key); if (q) { v = atof(q + strlen(key)); *found = 1; } }
    fclose(f); return v;
}
static double log_sum(const char *path, const char *key) {   /* the step lines' values summed (a window may draw no faceless ray) */
    FILE *f = fopen(path, "r"); if (!f) return NAN;
    char ln[8192]; double v = 0; int n = 0;
    while (fgets(ln, sizeof ln, f)) { const char *l = ln; if (l[0] == 't' && l[1] == '+') { const char *e = strstr(l, "s "); if (e) l = e + 2; }   /* (a "t+12.3s " prefix) */
        const char *q = strncmp(l, "step ", 5) ? nullptr : strstr(l, key); if (q) { v += atof(q + strlen(key)); n++; } }
    fclose(f); return n ? v : NAN;
}
static double log_first(const char *path, const char *key) {
    FILE *f = fopen(path, "r"); if (!f) return NAN;
    char ln[8192]; double v = NAN;
    while (fgets(ln, sizeof ln, f)) { const char *q = strncmp(ln, "step ", 5) ? nullptr : strstr(ln, key); if (q) { v = atof(q + strlen(key)); break; } }   /* (the step lines only) */
    fclose(f); return v;
}

int main(int argc, char **argv) {
    test_map();
    if (argc != 3) { fprintf(stderr, "usage: test_ink_mil_train <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    setenv("UFSM_STARTUP_TIMES", "0", 1);   /* (as run_suite: no "t+N.Ns" prefix on train's lines, which the '^memory:' / '^step' greps read; by hand = the suite) */
    char d[] = "/tmp/test_inkmil_XXXXXX"; if (!mkdtemp(d)) return 1;
    const int FN = getenv("TIMT_N") ? atoi(getenv("TIMT_N")) : 256;   /* TIMT_N / TIMT_KEEP=1 (benches): a larger fixture, kept with its MIL source */
    /* (FX_FAINT .375,.625: papyrus below the sources' iso 100 for z in [96, 160): the segment's rows there have no open face, a non-empty
       faceless set, so MIL mode 1 (the faceless rays) compares real losses below, not two zeros (lead 2026-11-20). CT 0 there instead
       (FX_CT_ZCUT) failed: the coarse probe rejects air windows, so no faceless ray was ever drawn, and the other sections starved) */
    CHECK(!sh("FX_FAINT=0.375,0.625 %s %s/fx %d 2 1 > %s/fx.log 2>&1", argv[2], d, FN, d), "fixture");
    char p[700]; snprintf(p, sizeof p, "%s/fx", d);
    CHECK(!write_segment(p, FN), "segment");
    snprintf(p, sizeof p, "%s/fx/sources-mil.json", d); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture-mil\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ink_weight\": 1, \"iso\": 100,\n"
               "  \"targets\": {\"ink\": {\"onfly\": {\"segments\": \"%s/fx/segs.json\", \"level\": 0, \"cache_dir\": \"%s/dcache\"}},\n"
               "               \"band\": {\"root\": \"%s/fx/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/fx/phase0\", \"key\": \"0\", \"level\": 0}}}}]}\n", d, d, d, d, d, d);
    fclose(f);
    const char *common = "--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --widths 16,32,64 --ink-head 1 --ink-dice 0.25";
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/a --steps 6 --ink-mil 1 --ink-mil-mode 0 %s > %s/a.log 2>&1", U, d, d, common, d), "train --ink-mil (%s/a.log)", d);
    snprintf(p, sizeof p, "%s/a.log", d); int fm = 0, fw = 0;
    const double mil = log_last(p, " ink mil ", &fm); (void)fw;
    CHECK(fm && isfinite(mil) && mil > 0 && mil < 5, "ink mil logged %d, value %g (%s)", fm, mil, p);
    {   /* windows with rays: the "(N% of windows" after the value */
        FILE *g = fopen(p, "r"); char ln[8192]; double best = 0, rays = 0;
        while (g && fgets(ln, sizeof ln, g)) { const char *q = strstr(ln, " ink mil "); if (!q) continue; const char *o = strchr(q, '(');
            if (o) { const double w = atof(o + 1); if (w > best) best = w; const char *rr = strstr(o, "windows, "); if (rr) { const double x = atof(rr + 9); if (x > rays) rays = x; } } }
        if (g) fclose(g);
        fprintf(stderr, "ink mil: last %.4f, at most %.0f%% of windows with rays, up to %.0f rays a window\n", mil, best, rays);
        CHECK(best > 0 && rays > 0, "no window drew rays (%s)", p);
    }
    {   /* ink-rayface (2026-11-20): the precomputed face-drop bits give mode 1's rays byte for byte (windows around supervised tiles,
           strides 1 and 2); a mode-1 run then reads them (no CT per window) and its first step's loss equals a run with the bits off */
        CHECK(!sh("%s ink-rayface %s/fx/sources-mil.json --threads 2 --verify 6 --P 64 --stride 1 > %s/rf1.log 2>&1", U, d, d), "ink-rayface --verify stride 1 (%s/rf1.log)", d);
        CHECK(!sh("%s ink-rayface %s/fx/sources-mil.json --threads 2 --verify 6 --P 64 --stride 2 --seed 7 > %s/rf2.log 2>&1", U, d, d), "ink-rayface --verify stride 2 (%s/rf2.log)", d);
        CHECK(!sh("grep -q '6 of 6 windows (P 64, stride 1) byte-identical' %s/rf1.log && grep -q '6 of 6 windows (P 64, stride 2) byte-identical' %s/rf2.log", d, d), "rayface rays differ (%s/rf1.log, rf2.log)", d);
        CHECK(!sh("grep -Eq 'faced pixels' %s/rf1.log && ! grep -q ' 0 faced pixels' %s/rf1.log", d, d), "the fixture faced no pixel (%s/rf1.log): mode 1 untested", d);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/m1 --steps 4 --lr 0 --muon-lr 0 --det 1 --ink-mil 1 --ink-mil-mode 1 %s > %s/m1.log 2>&1", U, d, d, common, d), "mode 1 with the bits (%s/m1.log)", d);
        CHECK(!sh("UFSM_INK_RAYFACE=0 UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/m0 --steps 4 --lr 0 --muon-lr 0 --det 1 --ink-mil 1 --ink-mil-mode 1 %s > %s/m0.log 2>&1", U, d, d, common, d), "mode 1 without (%s/m0.log)", d);
        snprintf(p, sizeof p, "%s/m1.log", d); const double l1 = log_sum(p, " ink mil "); snprintf(p, sizeof p, "%s/m0.log", d); const double l0 = log_sum(p, " ink mil ");
        fprintf(stderr, "ink-rayface: mode 1 ink mil (4 steps summed) %.6f with the bits, %.6f with the face search\n", l1, l0);
        CHECK(isfinite(l1) && l1 > 0 && l1 == l0, "mode 1 ink mil with the bits %g vs the face search %g (0: no faceless ray drawn)", l1, l0);
        /* (2026-11-20 inksurf) a version-2 mode-0 list (every pixel + its index) serves mode 1 through the face bits: the same loss again */
        CHECK(!sh("%s ink-rayface %s/fx/sources-mil.json --threads 2 --rays 2 --rays-mode 0 > %s/rfm0.log 2>&1 && grep -q 'ray lists (stride 2, mode 0)' %s/rfm0.log", U, d, d, d), "ink-rayface --rays-mode 0 (%s/rfm0.log)", d);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/m2 --steps 4 --lr 0 --muon-lr 0 --det 1 --ink-mil 1 --ink-mil-mode 1 %s > %s/m2.log 2>&1", U, d, d, common, d), "mode 1 from the mode-0 list (%s/m2.log)", d);
        snprintf(p, sizeof p, "%s/m2.log", d); const double l2 = log_sum(p, " ink mil ");
        CHECK(isfinite(l2) && l2 > 0 && l2 == l1, "mode 1 ink mil from the mode-0 list %g vs %g", l2, l1);
    }
    /* the planner keeps the MIL buffers free (2026-11-15 try7 one card: allocated lazily after the plan, the first MIL window's cudaMalloc
       failed on the full card): the "keeps" line, and the buffers never outgrow it; again at a small ray cap the windows exceed (the cap's
       sizes are the ones reached) */
    /* (the planner runs without UFSM_GRAD_MX8, which picks its mode by hand: these two runs leave it out) */
    CHECK(!sh("%s train %s/fx/sources-mil.json --out %s/d --steps 3 --ink-mil 1 --ink-mil-mode 0 %s > %s/d.log 2>&1", U, d, d, common, d), "train --ink-mil, planner (%s/d.log)", d);
    CHECK(!sh("grep -q '^memory: --ink-mil keeps [0-9.]* GB free for its buffers (65536 rays' %s/d.log && ! grep -q 'ink mil: WARNING its buffers reached' %s/d.log", d, d),
          "no MIL reservation line, or the buffers outgrew it (%s/d.log)", d);
    CHECK(!sh("%s train %s/fx/sources-mil.json --out %s/c --steps 4 --ink-mil 1 --ink-mil-mode 0 --ink-mil-max 64 %s > %s/c.log 2>&1", U, d, d, common, d), "train --ink-mil-max 64 (%s/c.log)", d);
    CHECK(!sh("grep -q '^memory: --ink-mil keeps [0-9.]* GB free for its buffers (64 rays' %s/c.log && ! grep -q 'ink mil: WARNING its buffers reached' %s/c.log", d, d),
          "--ink-mil-max 64: no reservation line, or the buffers outgrew it (%s/c.log)", d);
    /* the on-the-fly ink and --ct-gpu-decode on the training card (2026-11-20 FR1: the plan took the card, every ink chunk and (on an
       invisible device) the CT decode ran on the CPU): the planner keeps the ink's budget free and plans after the decoder's buffers exist */
    CHECK(!sh("UFSM_MEM_DRY=1 %s train %s/fx/sources-mil.json --out %s/e --steps 2 --ct-gpu-decode 0 --ink-gpu 0 %s > %s/e.log 2>&1", U, d, d, common, d), "dry plan (%s/e.log)", d);
    CHECK(!sh("grep -q '^memory: the on-the-fly ink on training GPU 0 keeps 1.34 GB free' %s/e.log && grep -q '^gpudec: [0-9.]* GB of device buffers on GPU 0' %s/e.log "
              "&& grep -q '^memory: [0-9.]* GB free on the device after the CT decode buffers' %s/e.log && ! grep -q '^DEFAULTED' %s/e.log", d, d, d, d),
          "no ink reservation / CT decode buffer lines, or a DEFAULTED line (%s/e.log)", d);
    CHECK(!sh("grep -q \"^memory: the on-the-fly ink's GPU budget [0-9.]* GB on GPU 0\" %s/d.log && ! grep -q '^DEFAULTED: inkfly' %s/d.log", d, d),
          "the ink's budget was not sized from the free memory, or chunks fell to the CPU (%s/d.log)", d);
    /* the loss bookkeeping: lr 0, same seed: the step-1 loss of --ink-mil 1 = that of --ink-mil 0 + the mil term */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/b1 --steps 1 --lr 0 --muon-lr 0 --ink-mil 1 --ink-mil-mode 0 %s > %s/b1.log 2>&1", U, d, d, common, d), "b1");
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/b0 --steps 1 --lr 0 --muon-lr 0 %s > %s/b0.log 2>&1", U, d, d, common, d), "b0");
    char p1[700], p0[700]; snprintf(p1, sizeof p1, "%s/b1.log", d); snprintf(p0, sizeof p0, "%s/b0.log", d);
    const double l1 = log_first(p1, " loss "), l0 = log_first(p0, " loss "); int fb = 0; const double m1 = log_last(p1, " ink mil ", &fb);
    fprintf(stderr, "lr 0 step 1: loss %.5f with mil %.5f, %.5f without\n", l1, m1, l0);
    CHECK(fb && fabs(l1 - (l0 + m1)) < 2e-3 + 1e-3 * fabs(l1), "loss %g != %g + %g", l1, l0, m1);
    /* serialised vs default (ops audit #9; 1bb0db8: the next window's rays landed under the current step's backward, a stream-ordering
       race): under --det the MIL run's step lines must not change when every launch completes in program order (UFSM_SERIALIZE=1) */
    if (!getenv("TIMT_NOSER")) {
        CHECK(!sh("%s train %s/fx/sources-mil.json --out %s/s0 --steps 4 --det 1 --ink-mil 1 --ink-mil-mode 0 %s > %s/s0.log 2>&1", U, d, d, common, d), "--det MIL run (%s/s0.log)", d);
        CHECK(!sh("UFSM_SERIALIZE=1 CUDA_LAUNCH_BLOCKING=1 %s train %s/fx/sources-mil.json --out %s/s1 --steps 4 --det 1 --ink-mil 1 --ink-mil-mode 0 %s > %s/s1.log 2>&1", U, d, d, common, d), "serialised MIL run (%s/s1.log)", d);
        CHECK(!sh("for x in s0 s1; do grep -E '^step' %s/$x.log | sed -E 's/ [0-9.]+ samp\\/s.*//; s/ (wait|gn) [0-9.%%]+//g' > %s/$x.txt; done; "
                  "[ $(grep -c '^step' %s/s0.txt) -ge 4 ] && cmp -s %s/s0.txt %s/s1.txt", d, d, d, d, d),
              "MIL training: default and UFSM_SERIALIZE=1 step lines differ (a stream-ordering race?): diff %s/s0.txt %s/s1.txt", d, d);
    }
    {   /* the stratified ink draw (2026-11-20; sampler-only, no GPU): an ink-only source drew ~97 % windows without an ink positive and dropped 4
           of 5 of them after the CT read (empty_keep 0.2). With --ink-pos-frac 0.5 (default) half the windows centre on a positive-label cell and
           none are dropped as empty: fewer rejections than the old draw (--ink-pos-frac -1), a startup line with the per-source cell counts,
           and a positive share of the known ink in a sane range (more than the old draw's, below 0.9) */
        snprintf(p, sizeof p, "%s/fx/sources-ink.json", d); FILE *g = fopen(p, "w"); if (!g) return 1;
        fprintf(g, "{\"sources\": [{\"name\": \"wind-fixture-ink\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ink_weight\": 1, \"iso\": 100,\n"
                   "  \"targets\": {\"ink\": {\"onfly\": {\"segments\": \"%s/fx/segs.json\", \"level\": 0, \"cache_dir\": \"%s/dcache\"}}}}]}\n", d, d, d, d);
        fclose(g);
        const char *so = "CUDA_VISIBLE_DEVICES= UFSM_SAMPLER_ONLY=40 UFSM_NCPU=4";
        for (int v = 0; v < 2; v++)
            CHECK(!sh("%s %s train %s/fx/sources-ink.json --out %s/so%d %s --single ink --ink-pos-frac %s > %s/so%d.log 2>&1", so, U, d, d, v, common, v ? "0.5" : "-1", d, v),
                  "sampler-only %d (%s/so%d.log)", v, d, v);
        double pr[2], rj[2], kn[2], ik[2]; int ok = 1;
        for (int v = 0; v < 2; v++) {
            char q[700]; snprintf(q, sizeof q, "%s/so%d.log", d, v); FILE *h = fopen(q, "r"); char ln[8192]; pr[v] = rj[v] = kn[v] = ik[v] = -1;
            while (h && fgets(ln, sizeof ln, h)) {
                const char *a = strstr(ln, "sampler-only: produced "); if (a) sscanf(a, "sampler-only: produced %lf rejected %lf", &pr[v], &rj[v]);
                const char *b = strstr(ln, " windows, known "); if (b) sscanf(b, " windows, known %lf ink %lf", &kn[v], &ik[v]);
            }
            if (h) fclose(h);
            ok &= pr[v] > 0 && rj[v] >= 0 && kn[v] > 0 && ik[v] >= 0;
        }
        const double f0 = ok ? ik[0] / kn[0] : -1, f1 = ok ? ik[1] / kn[1] : -1;
        fprintf(stderr, "ink draw: old %g kept / %g rejected, positive share %.3f; stratified %g / %g, %.3f\n", pr[0], rj[0], f0, pr[1], rj[1], f1);
        CHECK(ok && rj[1] < rj[0] && f1 > f0 && f1 < 0.9, "stratified draw: rejected %g vs %g, positive share %.3f vs %.3f (%s/so0.log, so1.log)", rj[1], rj[0], f1, f0, d);
        CHECK(!sh("grep -q 'sampler: wind-fixture-ink: ink draw cells: [0-9]* supervised, [1-9][0-9]* with positive labels$' %s/so1.log", d), "no per-source draw-cell line (%s/so1.log)", d);
    }
    {   /* train --ink-surf (2026-11-20, DESIGN "inksurf"): the surface head beside the voxel ink losses, its patch from the mode-2 rays (the MIL keeps
           its mode-1 subset): the log's "ink surf" finite and positive on windows with a patch, the unmasked share > 0; RUN/last.ckpt.isf written
           and taken by --resume; at lr 0 (all three; MIL mode 0, rays on the fixture) the step lines' other losses equal a run without it (the head adds only its own term;
           the MIL's rays from the mode-2 list are mode 0's) */
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/sf --steps 6 --ckpt-every 3 --ink-mil 1 --ink-mil-mode 1 --ink-surf 1 --isf-P 32 --isf-step 2 --isf-k 2 %s > %s/sf.log 2>&1",
                  U, d, d, common, d), "train --ink-surf (%s/sf.log)", d);
        snprintf(p, sizeof p, "%s/sf.log", d); int fs = 0; const double sl = log_last(p, " ink surf ", &fs);
        FILE *g = fopen(p, "r"); char ln[8192]; double win = 0, um = 0;
        while (g && fgets(ln, sizeof ln, g)) { const char *q = strstr(ln, " ink surf "); if (!q) continue; const char *o = strchr(q, '(');
            if (o) { const double w = atof(o + 1); if (w > win) win = w; const char *uu = strstr(o, "patch "); if (uu) { const double x = atof(uu + 6); if (x > um) um = x; } } }
        if (g) fclose(g);
        fprintf(stderr, "ink surf: last %.4f, up to %.0f%% of windows with a patch, up to %.0f%% unmasked\n", sl, win, um);
        CHECK(fs && isfinite(sl) && sl > 0 && sl < 5 && win > 0 && um > 0, "ink surf logged %d value %g, windows %g%%, unmasked %g%% (%s)", fs, sl, win, um, p);
        snprintf(p, sizeof p, "%s/sf/last.ckpt.isf", d); CHECK(!access(p, F_OK), "no %s", p);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/sf2 --steps 2 --resume %s/sf/last.ckpt --ink-mil 1 --ink-mil-mode 1 --ink-surf 1 --isf-P 32 --isf-step 2 --isf-k 2 %s > %s/sf2.log 2>&1",
                  U, d, d, d, common, d), "--resume with the head (%s/sf2.log)", d);
        CHECK(!sh("grep -q 'ink surf: head from .*last.ckpt.isf (step 6)' %s/sf2.log", d), "--resume did not take the surface head (%s/sf2.log)", d);
        /* ckpt-avg (11-21): averages the .isf head beside the trunk: parameters = the mean of sf and pert (sf with every head parameter
           + i % 7 / 8, Adam m + 1: sf2's two resumed steps leave the head unchanged), header and Adam moments those of the last input; an
           input without a sidecar next to one with is an error */
        {
            char pa[1024], pb[1024]; snprintf(pa, sizeof pa, "%s/sf/last.ckpt.isf", d); snprintf(pb, sizeof pb, "%s/pert.ckpt.isf", d);
            FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "wb"); long sa = 0; int ok = fa && fb;
            if (ok) { fseek(fa, 0, SEEK_END); sa = ftell(fa); rewind(fa); }
            unsigned char *A = ok && sa > 20 ? malloc(sa) : nullptr; ok = A && fread(A, 1, sa, fa) == (size_t)sa;
            if (ok) {
                int hdr; memcpy(&hdr, A, 4); const long hl = hdr == 0x31465349 ? 16 : 20, np = (sa - hl) / 12;
                for (long i = 0; i < 2 * np; i++) { float x; memcpy(&x, A + hl + 4 * i, 4); x += i < np ? (float)(i % 7) / 8 : 1.f; memcpy(A + hl + 4 * i, &x, 4); }
                ok = fwrite(A, 1, sa, fb) == (size_t)sa;
            }
            free(A); if (fa) fclose(fa); if (fb && fclose(fb)) ok = 0;
            CHECK(ok, "cannot write %s", pb);
        }
        CHECK(!sh("cp %s/sf/last.ckpt %s/pert.ckpt && %s ckpt-avg %s/avg.ckpt %s/sf/last.ckpt %s/pert.ckpt > %s/avg.log 2>&1", d, d, U, d, d, d, d), "ckpt-avg (%s/avg.log)", d);
        {
            char pa[1024], pb[1024], po[1024]; snprintf(pa, sizeof pa, "%s/sf/last.ckpt.isf", d); snprintf(pb, sizeof pb, "%s/pert.ckpt.isf", d);
            snprintf(po, sizeof po, "%s/avg.ckpt.isf", d);
            FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb"), *fo = fopen(po, "rb"); long sa = 0, sb = 0, so = 0;
            if (fa) { fseek(fa, 0, SEEK_END); sa = ftell(fa); rewind(fa); } if (fb) { fseek(fb, 0, SEEK_END); sb = ftell(fb); rewind(fb); }
            if (fo) { fseek(fo, 0, SEEK_END); so = ftell(fo); rewind(fo); }
            CHECK(fa && fb && fo && sa == sb && sb == so && sa > 16, "ckpt-avg: no / mis-sized %s (%ld %ld %ld)", po, sa, sb, so);
            if (fa && fb && fo && sa == sb && sb == so && sa > 16) {
                unsigned char *A = malloc(sa), *B = malloc(sa), *O = malloc(sa);
                const int rd = fread(A, 1, sa, fa) == (size_t)sa && fread(B, 1, sa, fb) == (size_t)sa && fread(O, 1, sa, fo) == (size_t)sa;
                int hdr; memcpy(&hdr, A, 4); const long hl = hdr == 0x31465349 ? 16 : 20, np = (sa - hl) / 12; long bad = 0, moved = 0;
                for (long i = 0; i < np; i++) {
                    float x, y, z; memcpy(&x, A + hl + 4 * i, 4); memcpy(&y, B + hl + 4 * i, 4); memcpy(&z, O + hl + 4 * i, 4);
                    bad += z != (float)(((double)x + y) / 2); moved += x != y;
                }
                CHECK(rd && !bad && moved > 0, "ckpt-avg .isf: %ld of %ld parameters not the mean (%ld differ between the inputs)", bad, np, moved);
                CHECK(!memcmp(O, B, hl) && !memcmp(O + hl + 4 * np, B + hl + 4 * np, 8 * np), "ckpt-avg .isf: header / moments not the last input's");
                free(A); free(B); free(O);
            }
            if (fa) fclose(fa); if (fb) fclose(fb); if (fo) fclose(fo);
        }
        CHECK(sh("cp %s/sf/last.ckpt %s/nohead.ckpt && %s ckpt-avg %s/avg2.ckpt %s/nohead.ckpt %s/pert.ckpt > %s/avg2.log 2>&1", d, d, U, d, d, d, d) &&
              !sh("grep -q '1 of 2 inputs have a .isf' %s/avg2.log", d), "ckpt-avg with one sidecar missing did not fail (%s/avg2.log)", d);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/sb1 --steps 2 --lr 0 --muon-lr 0 --det 1 --ink-mil 1 --ink-mil-mode 0 --ink-surf 1 --isf-lr 0 --isf-P 32 --isf-step 2 --isf-k 2 %s > %s/sb1.log 2>&1", U, d, d, common, d), "sb1");
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/sb0 --steps 2 --lr 0 --muon-lr 0 --det 1 --ink-mil 1 --ink-mil-mode 0 %s > %s/sb0.log 2>&1", U, d, d, common, d), "sb0");
        CHECK(!sh("for x in sb0 sb1; do grep -E '^(t\\+[0-9.]+s )?step ' %s/$x.log | sed -E 's/^t\\+[0-9.]+s //; s/ ink surf [^(]*\\([^)]*\\)//; s/ [0-9.]+ samp\\/s.*//; s/ (wait|gn) [0-9.%%]+//g' > %s/$x.txt; done; cmp -s %s/sb0.txt %s/sb1.txt", d, d, d, d),
              "the step losses differ with --ink-surf at lr 0 (%s/sb0.txt, sb1.txt)", d);
        /* (S3, 2026-11-21) a "surf_only" source (teacher maps): soft patch labels, its BCE logged apart (" teach "), no voxel ink and no MIL
           ray from it; a depth-5 head from the depth-4 sf head (--isf-init, 12 of 15 convs); --isf-pvox 2: the fixture's label px is one
           window voxel, so every 2nd label px (" step 2.0") */
        {   char sp[700]; snprintf(sp, sizeof sp, "%s/fx/sources-teach.json", d); FILE *tf = fopen(sp, "w");
            if (tf) { fprintf(tf, "{\"sources\": [{\"name\": \"wind-fixture-teach\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"surf_only\": 1, \"surf_weight\": 0.5, \"iso\": 100,\n"
                              "  \"targets\": {\"ink\": {\"onfly\": {\"segments\": \"%s/fx/segs.json\", \"level\": 0, \"cache_dir\": \"%s/dcache_t\"}},\n"
                              "               \"band\": {\"root\": \"%s/fx/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/fx/phase0\", \"key\": \"0\", \"level\": 0}}}}]}\n", d, d, d, d, d, d); fclose(tf); }
            CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s --out %s/st --steps 4 --ink-mil 1 --ink-mil-mode 1 --ink-surf 1 --isf-P 32 --isf-step 2 --isf-k 2 --isf-depth 5 --isf-pvox 2 --isf-init %s/sf/last.ckpt.isf %s > %s/st.log 2>&1",
                      U, sp, d, d, common, d), "train with a surf_only source (%s/st.log)", d);
            CHECK(!sh("grep -q 'ink surf: head from --isf-init .*(depth 4 -> 5: 12 of 15 convs taken' %s/st.log && grep -Eq '^step .* step 2.0 teach [0-9.]+ \\(100%% of windows\\)' %s/st.log "
                      "&& grep -Eq '^step .* ink bce 0.0000 \\(0%% of windows\\)' %s/st.log && grep -Eq '^step .* ink mil 0.0000 \\(0%% of windows, 0 rays\\)' %s/st.log", d, d, d, d),
                  "surf_only source: --isf-init mapping, teach BCE on every window, no voxel ink / MIL (%s/st.log)", d);
        }
        /* predict --surf (2026-11-21): the trunk features in 32-plane z-slabs (training's unet_head_feat_z) give the map of the whole-window
           feature copy (UFSM_SURF_WHOLE=1, the old path, 10.3 GB at window 544) byte for byte, every patch run (an unplaced patch prints DEFAULTED, which sp0 must not) */
        const char *pa = "--um 9.6 --level 0 --box 0,0,0,256,256,256 --shard 128 --levels 2 --threads 2 --q 0 --no-out 1 --surf-P 32 --surf-halo 4 --surf-k 2 --window 128";
        CHECK(!sh("mkdir -p %s/sp0 %s/sp1 && %s predict %s/sf/last.ckpt %s/fx ct %s/pv0 --axis %s/fx/axis.json --surf %s/fx/seg.tifxyz --surf-out %s/sp0 %s > %s/sp0.log 2>&1",
                  d, d, U, d, d, d, d, d, d, pa, d), "predict --surf (%s/sp0.log)", d);
        CHECK(!sh("UFSM_SURF_WHOLE=1 %s predict %s/sf/last.ckpt %s/fx ct %s/pv1 --axis %s/fx/axis.json --surf %s/fx/seg.tifxyz --surf-out %s/sp1 %s > %s/sp1.log 2>&1",
                  U, d, d, d, d, d, d, pa, d), "predict --surf, whole window (%s/sp1.log)", d);
        CHECK(!sh("grep -q 'DEFAULTED: predict --surf: UFSM_SURF_WHOLE' %s/sp1.log && ! grep -q DEFAULTED %s/sp0.log", d, d), "DEFAULTED lines (%s/sp0.log, sp1.log)", d);
        CHECK(!sh("grep -q 'patches run' %s/sp0.log && cmp %s/sp0/seg.tifxyz.tif %s/sp1/seg.tifxyz.tif", d, d, d),
              "slab vs whole-window maps differ or patches unrun (%s/sp0, sp1)", d);
        /* (2026-11-21) at level 1 (2 level-0 voxels per window voxel) the map is still written on the mesh's canvas (what ink-score reads):
           the level-0 map's size; the 2x grid read as canvas px was the S1 0841 / P4 surface-score bug (AUC 0.03 / 0.30) */
        CHECK(!sh("mkdir -p %s/sp2 && %s predict %s/sf/last.ckpt %s/fx ct %s/pv2 --axis %s/fx/axis.json --surf %s/fx/seg.tifxyz --surf-out %s/sp2 --um 19.2 --level 1 "
                  "--box 0,0,0,128,128,128 --shard 128 --levels 2 --threads 2 --q 0 --no-out 1 --surf-P 32 --surf-halo 4 --surf-k 2 --window 96 > %s/sp2.log 2>&1", d, U, d, d, d, d, d, d, d),
              "predict --surf at level 1 (%s/sp2.log)", d);
        {   char q0[700], q2[700]; snprintf(q0, sizeof q0, "%s/sp0/seg.tifxyz.tif", d); snprintf(q2, sizeof q2, "%s/sp2/seg.tifxyz.tif", d);
            int w0 = 0, h0 = 0, w2 = 0, h2 = 0; uint8_t *m0 = img_read_u8(q0, &w0, &h0), *m2 = img_read_u8(q2, &w2, &h2); size_t nz = 0;
            if (m2) for (size_t i = 0; i < (size_t)w2 * h2; i++) nz += m2[i] > 0;
            fprintf(stderr, "predict --surf: level-0 map %d x %d, level-1 map %d x %d (%.0f%% run px)\n", w0, h0, w2, h2, m2 ? 100.0 * nz / ((double)w2 * h2) : 0);
            CHECK(m0 && m2 && w0 == w2 && h0 == h2 && nz > (size_t)w2 * h2 / 2, "level-1 map %d x %d vs the canvas %d x %d (%s)", w2, h2, w0, h0, q2);
            free(m0); free(m2); }
        /* --surf-overlap (2026-11-21, 1447's hard patch seams): 0 = the hard tiling byte for byte (sp2); 0.5 = patches every half core,
           feathered: it runs, says so, and covers every pixel the tiling covers (its extra edge patches in windows without CT stay unrun: DEFAULTED there is expected) */
        {   const char *L1 = "--um 19.2 --level 1 --box 0,0,0,128,128,128 --shard 128 --levels 2 --threads 2 --q 0 --no-out 1 --surf-P 32 --surf-halo 4 --surf-k 2 --window 96";
            CHECK(!sh("mkdir -p %s/sp4 %s/sp3 && %s predict %s/sf/last.ckpt %s/fx ct %s/pv4 --axis %s/fx/axis.json --surf %s/fx/seg.tifxyz --surf-out %s/sp4 %s --surf-overlap 0 > %s/sp4.log 2>&1 && "
                      "cmp %s/sp2/seg.tifxyz.tif %s/sp4/seg.tifxyz.tif", d, d, U, d, d, d, d, d, d, L1, d, d, d), "--surf-overlap 0 is not the hard tiling's map (%s/sp4)", d);
            CHECK(!sh("%s predict %s/sf/last.ckpt %s/fx ct %s/pv3 --axis %s/fx/axis.json --surf %s/fx/seg.tifxyz --surf-out %s/sp3 %s --surf-overlap 0.5 --surf-blend cosine > %s/sp3.log 2>&1 && "
                      "grep -q 'surf-overlap 0.50: patches every 12 px, cosine feather over 12 px' %s/sp3.log", U, d, d, d, d, d, d, L1, d, d),
                  "predict --surf --surf-overlap 0.5 (%s/sp3.log)", d);
            char q2[700], q3[700]; snprintf(q2, sizeof q2, "%s/sp2/seg.tifxyz.tif", d); snprintf(q3, sizeof q3, "%s/sp3/seg.tifxyz.tif", d);
            int w2 = 0, h2 = 0, w3 = 0, h3 = 0; uint8_t *m2 = img_read_u8(q2, &w2, &h2), *m3 = img_read_u8(q3, &w3, &h3); size_t lost = 0, n2 = 0, same = 0;
            if (m2 && m3 && w2 == w3 && h2 == h3) for (size_t i = 0; i < (size_t)w2 * h2; i++) { n2 += m2[i] > 0; lost += m2[i] > 0 && !m3[i]; same += m2[i] == m3[i]; }
            fprintf(stderr, "predict --surf-overlap 0.5: %zu of %zu tiled px kept, %.0f%% of px equal to the tiling\n", n2 - lost, n2, m2 ? 100.0 * same / ((double)w2 * h2) : 0);
            CHECK(m2 && m3 && w2 == w3 && h2 == h3 && n2 > 0 && !lost && same < (size_t)w2 * h2, "overlap map %d x %d vs %d x %d, %zu tiled px lost (%s)", w3, h3, w2, h2, lost, q3);
            free(m2); free(m3); }
    }
    if (getenv("TIMT_KEEP")) fprintf(stderr, "kept %s (sources %s/fx/sources-mil.json)\n", d, d);
    else if (!fails) sh("rm -rf %s", d);
    fprintf(stderr, "%s\n", fails ? "test_ink_mil_train: FAILED" : "test_ink_mil_train: ok");
    return fails != 0;
}
