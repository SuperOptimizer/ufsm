/* predict is byte-identical run to run (GroupNorm statistics as exact fixed-point sums in a fixed block order: src/gnacc.cuh; the CT
   prefetch on or off). Small random 2-level nets (legacy unrounded statistics with MX-fp8 activations, stored statistics with MX-fp8
   and with MX-fp4 activations) on a 256^3 CT fixture, a box at 16 with window 96 halo 8 shard 128 (64 windows, repeated across shards,
   CT edges): three runs each (prefetch on twice, off once) give the same bytes; and predict --feather 16 run twice. Before the fix the
   double atomics of float block partials made most of these windows differ between runs.
   Usage: test_predict_det <ufsm binary> */
#include "nn.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
enum { N = 256 };

static int predict(const char *ufsm, const char *dir, const char *ck, const char *name, const char *env, const char *geo) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "%s %s predict %s/%s %s ct %s/%s --um 1 --level 0 --ema 1 --threads 2 --levels 1 --q 0 %s >> %s/log-%s.txt 2>&1",
             env, ufsm, dir, ck, dir, dir, name, geo, dir, name);
    return system(cmd);
}
static uint8_t *load(const char *dir, const char *name, int M) {
    char p[600]; snprintf(p, sizeof p, "%s/%s", dir, name);
    store *s = store_open(p); z3 *z = s ? z3_open(s, "1", nullptr) : nullptr;
    uint8_t *v = malloc((size_t)M * M * M); const int64_t o[3] = {0, 0, 0}, n[3] = {M, M, M};
    if (!z || !v || z3_read(z, o, n, v, 2)) { fprintf(stderr, "cannot read %s: %s\n", p, z3_error()); free(v); v = nullptr; }
    if (z) z3_close(z);
    if (s) store_close(s);
    return v;
}
static size_t ndiff(const uint8_t *a, const uint8_t *b, size_t n) { size_t d = 0; for (size_t i = 0; i < n; i++) d += a[i] != b[i]; return d; }

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_predict_det <ufsm binary>\n"); return 2; }
    char dir[] = "/tmp/test_pdet_XXXXXX"; if (!mkdtemp(dir)) return 1;
    { const int64_t sh[3] = {N, N, N}; const double um = 1; char p[512]; uint8_t *d = malloc((size_t)N * N * N);
      for (size_t i = 0; i < (size_t)N * N * N; i++) {   /* the feather test's fixture: ramps, a contrast step, noise */
          const int z = (int)(i / (N * N)), y = (int)(i / N % N), x = (int)(i % N);
          d[i] = (uint8_t)(20 + 0.5 * z + 0.2 * y + 30 * sin(0.11 * x + 0.07 * y) * cos(0.05 * z) * (1 + (x > 140)) + (int)((i * 2654435761u) >> 28));
      }
      snprintf(p, sizeof p, "%s/ct/1", dir); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr);
      int bad = !w;
      for (int a = 0; a < 2 && !bad; a++) for (int b = 0; b < 2 && !bad; b++) for (int c = 0; c < 2 && !bad; c++) {
          uint8_t *s = malloc(128 * 128 * 128);
          for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(s + ((size_t)z * 128 + y) * 128, d + ((size_t)(a * 128 + z) * N + b * 128 + y) * N + c * 128, 128);
          bad = z3w_write_shard(w, a, b, c, s, 2); free(s);
      }
      CHECK(!bad && !z3w_close(w), "CT fixture");
      snprintf(p, sizeof p, "%s/ct", dir); CHECK(!z3w_write_group(p, &um, 1, "ct", nullptr), "CT group"); free(d); }
    if (nn_init(0)) { fprintf(stderr, "test_predict_det: no GPU\n"); return 1; }
    static const struct { const char *ck; int mx4, stored; } nets[3] = {{"leg8.ckpt", 0, 0}, {"st8.ckpt", 0, 1}, {"st4.ckpt", 1, 1}};
    for (int k = 0; k < 3; k++) {
        unet_cfg cfg = {0}; cfg.nlev = 2; cfg.widths[0] = 32; cfg.widths[1] = 64; cfg.cin = 4; cfg.cout = 1; cfg.G = 8; cfg.down_norm = 1;
        unet *u = unet_create(&cfg); unet_init(u, 11 + k); unet_ema(u, 0.f);
        char ck[512], rt[600]; snprintf(ck, sizeof ck, "%s/%s", dir, nets[k].ck);
        snprintf(rt, sizeof rt, "{\"runtime\":{\"version\":1,\"train_window\":96,\"prec\":1,\"f16\":1,\"act_mx4\":%d,\"act_mx8\":1,\"grad_mx8\":1,\"input_mx\":1,\"input_prec\":8,"
                 "\"gn_stored\":%d,\"policy\":\"\",\"optimizer\":\"muon\"}}", nets[k].mx4, nets[k].stored);
        CHECK(!unet_save(u, ck, 1, rt), "save %s", nets[k].ck);
        unet_free(u);
    }
    const char *geo = "--box 16,16,16,240,240,240 --window 96 --halo 8 --shard 128";
    for (int k = 0; k < 3; k++) {
        char a[64], b[64], c[64]; snprintf(a, sizeof a, "%d-a", k); snprintf(b, sizeof b, "%d-b", k); snprintf(c, sizeof c, "%d-c", k);
        CHECK(predict(argv[1], dir, nets[k].ck, a, "", geo) == 0 && predict(argv[1], dir, nets[k].ck, b, "", geo) == 0 && predict(argv[1], dir, nets[k].ck, c, "UFSM_PRED_CTPRE=0", geo) == 0,
              "%s predicts (%s/log-%s.txt)", nets[k].ck, dir, a);
        uint8_t *va = load(dir, a, 240), *vb = load(dir, b, 240), *vc = load(dir, c, 240);
        if (va && vb && vc) {
            const size_t n = (size_t)240 * 240 * 240, db = ndiff(va, vb, n), dc = ndiff(va, vc, n); size_t nz = 0;
            for (size_t i = 0; i < n; i++) nz += va[i] != 0;
            CHECK(nz > n / 4, "%s: only %zu nonzero voxels", nets[k].ck, nz);
            CHECK(!db, "%s: %zu of %zu voxels differ between two runs", nets[k].ck, db, n);
            CHECK(!dc, "%s: %zu of %zu voxels differ with the CT prefetch off", nets[k].ck, dc, n);
            fprintf(stderr, "test_predict_det: %s %zu voxels, run diffs %zu, prefetch-off diffs %zu\n", nets[k].ck, n, db, dc);
        } else CHECK(0, "%s outputs", nets[k].ck);
        free(va); free(vb); free(vc);
    }
    /* --feather (the ink default's blend) on the legacy net, twice */
    CHECK(predict(argv[1], dir, "leg8.ckpt", "fa", "", "--window 160 --halo 16 --shard 128 --feather 16") == 0 &&
          predict(argv[1], dir, "leg8.ckpt", "fb", "", "--window 160 --halo 16 --shard 128 --feather 16") == 0, "--feather predicts (%s/log-fa.txt)", dir);
    { uint8_t *fa = load(dir, "fa", N), *fb = load(dir, "fb", N);
      if (fa && fb) { const size_t d = ndiff(fa, fb, (size_t)N * N * N); CHECK(!d, "--feather: %zu voxels differ between two runs", d); }
      else CHECK(0, "--feather outputs");
      free(fa); free(fb); }
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_predict_det: %d FAILED (kept %s)\n", fails, dir); return 1; }
    printf("test_predict_det: ok\n");
    return 0;
}
