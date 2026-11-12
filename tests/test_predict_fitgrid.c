/* predict --fit-zarr with a box off the window grid (--grid-origin): windows are anchored on the global grid, shards / field slabs at
   the box, so a window's interior can cover two shards; its cells are clipped to the tile's shard (regression 2026-11-11: unclipped,
   the cells of a window across a field-slab boundary were written past the slab: segfault in ffw_main on the wbench DLV -1 boxes with
   640 / 64 latent windows). A constant-logit net (zero head, distinct biases; test_predict_unwrap2's) on a 256 x 128 x 128 CT:
   --box 32,0,0,192,128,128 --grid-origin 0,0,0, --window 96 --halo 16 (stride 64: z phase 32), --fit-h 1 -> 192 cells in z = two
   slabs of 128, the window interior 96 .. 160 across them. Every cell must be known with the constant phase, and the box on the grid
   (z 64) must give the same planes. Also off the grid: --fit-h 4, y / x, all axes, and level 1 across a slab: every cell known. Usage: test_predict_fitgrid <ufsm binary> */
#include "fit.h"
#include "nn.h"
#include "unet.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static const float bias[11] = {2.0f, 0.0f, -2.0f, 1.0f, 0.5f, -0.5f, 0.3f, -0.3f, 0.7f, 1.5f, -1.5f};
static int pcode(float l) { return (int)lrintf(128 + 127 * (2 / (1 + expf(-l)) - 1)); }   /* pc / ps of a constant logit */

static int predictx(const char *ufsm, const char *dir, const char *ck, const char *out, const char *box, const char *extra) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "%s predict %s %s ct %s/pred-%s --um 1 --box %s --grid-origin 0,0,0 --shard 128 --ema 1 --threads 2 --levels 1 "
             "--fit-zarr %s --fit-q 0 --fit-q-phase 0 --no-out 1 %s > %s/log-%s.txt 2>&1", ufsm, ck, dir, dir, strrchr(out, '/') + 1, box, out, extra, dir, strrchr(out, '/') + 1);
    return system(cmd);
}
static int predict(const char *ufsm, const char *dir, const char *ck, const char *out, const char *box) {
    return predictx(ufsm, dir, ck, out, box, "--level 0 --window 96 --halo 16 --fit-h 1");
}
/* every cell of the fields known, with the constant phase */
static void check_known(const char *out, const char *what) {   /* (the stores hold --fit-h cells: loaded unpooled) */
    fit_fields_buf a;
    if (fit_fields_load(out, 0, 0, 1, &a)) { CHECK(0, "%s: load the fields (%s)", what, out); return; }
    const size_t n = (size_t)a.shape[0] * a.shape[1] * a.shape[2]; long unk = 0, badp = 0;
    for (size_t i = 0; i < n; i++) { if (!a.a[9][i]) unk++; else if (abs(a.a[0][i] - pcode(bias[0])) > 1 || abs(a.a[1][i] - pcode(bias[1])) > 1) badp++; }
    CHECK(!unk && !badp, "%s: %ld of %zu cells unknown, %ld off the constant phase", what, unk, n, badp);
    fit_fields_free(&a);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_predict_fitgrid <ufsm binary>\n"); return 2; }
    char dir[] = "/tmp/test_pfg_XXXXXX"; if (!mkdtemp(dir)) return 1;   /* (small: a 256 x 128 x 128 CT, a ~0.4 MB checkpoint) */
    /* the CT fixture: material everywhere, two shards in z */
    { const int64_t sh[3] = {256, 128, 128}; const double um = 1; char p[512]; uint8_t *d = malloc(128 * 128 * 128);
      for (int i = 0; i < 128 * 128 * 128; i++) d[i] = (uint8_t)(60 + (i * 7) % 150);
      snprintf(p, sizeof p, "%s/ct/1", dir); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr);
      CHECK(w && !z3w_write_shard(w, 0, 0, 0, d, 2) && !z3w_write_shard(w, 1, 0, 0, d, 2) && !z3w_close(w), "CT fixture");
      {   const int64_t s2[3] = {256, 64, 64}; uint8_t *d2 = calloc(128 * 128 * 128, 1);   /* level 1 (2 um): 256 x 64 x 64, two shards in z (its boxes cross a field slab) */
          for (int z = 0; z < 128; z++) for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) d2[((size_t)z * 128 + y) * 128 + x] = (uint8_t)(60 + ((z * 64 + y) * 64 + x) * 7 % 150);
          snprintf(p, sizeof p, "%s/ct/2", dir); z3w *w2 = z3w_create(p, s2, 128, 0, 0, nullptr);
          CHECK(w2 && !z3w_write_shard(w2, 0, 0, 0, d2, 2) && !z3w_write_shard(w2, 1, 0, 0, d2, 2) && !z3w_close(w2), "CT level 1"); free(d2); }
      snprintf(p, sizeof p, "%s/ct", dir); const double ums[2] = {1, 2}; CHECK(!z3w_write_group(p, ums, 2, "ct", nullptr), "CT group"); free(d); (void)um; }
    /* the net: zero head weights, distinct biases; EMA = live weights */
    if (nn_init(0)) { fprintf(stderr, "test_predict_fitgrid: no GPU\n"); return 1; }
    unet_cfg cfg = {0}; cfg.nlev = 2; cfg.widths[0] = 32; cfg.widths[1] = 64; cfg.cin = 4; cfg.cout = 11; cfg.G = 8; cfg.down_norm = 1;
    unet *u = unet_create(&cfg); unet_init(u, 7);
    const size_t np = unet_nparams(u); size_t hw, hb; unet_head_offsets(u, &hw, &hb);
    float *h = malloc(np * 4); nn_d2h(h, unet_live_params(u), np * 4);
    for (size_t i = hw; i < hb; i++) h[i] = 0.f;
    for (int c = 0; c < 11; c++) h[hb + c] = bias[c];
    nn_h2d((float *)unet_live_params(u), h, np * 4); unet_ema(u, 0.f); free(h);
    char ck2[512]; snprintf(ck2, sizeof ck2, "%s/u2.ckpt", dir);
    CHECK(!unet_save(u, ck2, 1, "{\"runtime\":{\"version\":1,\"train_window\":96,\"prec\":1,\"f16\":1,\"act_mx4\":0,\"act_mx8\":1,\"grad_mx8\":1,\"input_mx\":1,\"input_prec\":8,\"gn_stored\":0,\"policy\":\"\",\"optimizer\":\"muon\"},\"unwrap\":{\"version\":2,\"density_scale\":4,\"density_code\":1}}"), "save unwrap2");
    unet_free(u);
    char oo[512], og[512]; snprintf(oo, sizeof oo, "%s/foff", dir); snprintf(og, sizeof og, "%s/fgrid", dir);
    CHECK(predict(argv[1], dir, ck2, oo, "32,0,0,192,128,128") == 0, "predict, box off the window grid (%s/log-foff.txt)", dir);
    CHECK(predict(argv[1], dir, ck2, og, "64,0,0,192,128,128") == 0, "predict, box on the window grid (%s/log-fgrid.txt)", dir);
    fit_fields_buf a, g;
    const int la = fit_fields_load(oo, 0, 0, 1, &a), lg = fit_fields_load(og, 0, 0, 1, &g);
    CHECK(!la && !lg, "load the fields");
    if (!la && !lg) {
        CHECK(a.shape[0] == 192 && a.shape[1] == 128 && a.shape[2] == 128, "field shape %d %d %d", a.shape[0], a.shape[1], a.shape[2]);
        const size_t pl = (size_t)a.shape[1] * a.shape[2], n = (size_t)a.shape[0] * pl;
        long unk = 0, badp = 0, diff = 0; int zfirst = -1;
        for (size_t i = 0; i < n; i++) {
            if (!a.a[9][i]) { unk++; if (zfirst < 0) zfirst = (int)(i / pl); }
            else if (abs(a.a[0][i] - pcode(bias[0])) > 1 || abs(a.a[1][i] - pcode(bias[1])) > 1) badp++;
        }
        CHECK(!unk, "%ld of %zu cells unknown (w 0), the first at z %d: cells of a window across the slab boundary at z 128 lost", unk, n, zfirst);
        CHECK(!badp, "%ld known cells off the constant phase (pc %d ps %d)", badp, pcode(bias[0]), pcode(bias[1]));
        if (g.shape[0] == a.shape[0]) for (int k = 0; k < 10; k++) for (size_t i = 0; i < n; i++) if (a.a[k][i] != g.a[k][i]) diff++;
        CHECK(g.shape[0] == a.shape[0] && !diff, "off-grid and on-grid boxes differ in %ld plane cells (constant logits, material everywhere)", diff);
        fit_fields_free(&a); fit_fields_free(&g);
    }
    /* more boxes off the window grid: --fit-h 4 (the wbench DLV -1 crash's), y and x, all three axes, and level 1 (2 um, windows 48 / 8) */
    static const struct { const char *name, *box, *extra; } X[] = {
        {"z-h4", "32,0,0,192,128,128", "--level 0 --window 96 --halo 16 --fit-h 4"},
        {"yx", "64,32,32,128,96,96", "--level 0 --window 96 --halo 16 --fit-h 1"},
        {"zyx-h4", "16,48,16,224,80,112", "--level 0 --window 96 --halo 16 --fit-h 4"},
        {"lv1", "32,8,24,192,56,40", "--level 1 --window 96 --halo 16 --fit-h 1"},   /* (192 cells in z: across the slab at 128) */
    };
    for (int k = 0; k < 4; k++) {
        char o2[512]; snprintf(o2, sizeof o2, "%s/f%s", dir, X[k].name);
        CHECK(predictx(argv[1], dir, ck2, o2, X[k].box, X[k].extra) == 0, "predict, box %s off the window grid (%s/log-f%s.txt)", X[k].box, dir, X[k].name);
        check_known(o2, X[k].name);
    }
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_predict_fitgrid: %d FAILED (kept %s)\n", fails, dir); return 1; }
    printf("test_predict_fitgrid: ok\n");
    return 0;
}
