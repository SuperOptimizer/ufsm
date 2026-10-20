/* predict --fit-fields / --fit-zarr reads the head layout from the checkpoint (regression: the fitter branch's predict pooled task
   unwrap2 checkpoints as task unwrap ones, the phase from channels 2-3 = unwrap2's normals). A small net with a zero head and distinct
   head biases (every logit constant: channel c = bias c), saved once as task unwrap2 and once as task unwrap, predicted on a 128^3
   CT fixture: unwrap2 fields take the phase from channels 0-1 and write the winding mod 4 planes (no recto), unwrap ones the phase from
   channels 2-3 and the recto plane; --tta 2 leaves the (constant) phase as it is. Usage: test_predict_unwrap2 <ufsm binary> */
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

static int predict(const char *ufsm, const char *dir, const char *ck, const char *out, const char *extra) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "%s predict %s %s ct %s/pred-%s --um 1 --level 0 --window 96 --halo 16 --shard 128 --ema 1 --threads 2 --levels 1 "
             "--fit-zarr %s --fit-h 4 %s > %s/log-%s.txt 2>&1", ufsm, ck, dir, dir, strrchr(out, '/') + 1, out, extra, dir, strrchr(out, '/') + 1);
    return system(cmd);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_predict_unwrap2 <ufsm binary>\n"); return 2; }
    char dir[] = "/tmp/test_pu2_XXXXXX"; if (!mkdtemp(dir)) return 1;   /* (small: a 128^3 CT, two ~0.4 MB checkpoints) */
    /* the CT fixture: material everywhere */
    /* (the checkpoints carry an embedded runtime: fp16 / MX-fp8 storage with an MX input, the default build's kernels) */
    { const int64_t sh[3] = {128, 128, 128}; const double um = 1; char p[512]; uint8_t *d = malloc(128 * 128 * 128);
      for (int i = 0; i < 128 * 128 * 128; i++) d[i] = (uint8_t)(60 + (i * 7) % 150);
      snprintf(p, sizeof p, "%s/ct/1", dir); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr);
      CHECK(w && !z3w_write_shard(w, 0, 0, 0, d, 2) && !z3w_close(w), "CT fixture");
      snprintf(p, sizeof p, "%s/ct", dir); CHECK(!z3w_write_group(p, &um, 1, "ct", nullptr), "CT group"); free(d); }
    /* the net: zero head weights, distinct biases; EMA = live weights */
    if (nn_init(0)) { fprintf(stderr, "test_predict_unwrap2: no GPU\n"); return 1; }
    unet_cfg cfg = {0}; cfg.nlev = 2; cfg.widths[0] = 32; cfg.widths[1] = 64; cfg.cin = 4; cfg.cout = 11; cfg.G = 8; cfg.down_norm = 1;
    unet *u = unet_create(&cfg); unet_init(u, 7);
    const size_t np = unet_nparams(u); size_t hw, hb; unet_head_offsets(u, &hw, &hb);
    float *h = malloc(np * 4); nn_d2h(h, unet_live_params(u), np * 4);
    for (size_t i = hw; i < hb; i++) h[i] = 0.f;
    for (int c = 0; c < 11; c++) h[hb + c] = bias[c];
    nn_h2d((float *)unet_live_params(u), h, np * 4); unet_ema(u, 0.f); free(h);
    char ck2[512], ck1[512]; snprintf(ck2, sizeof ck2, "%s/u2.ckpt", dir); snprintf(ck1, sizeof ck1, "%s/u1.ckpt", dir);
    CHECK(!unet_save(u, ck2, 1, "{\"runtime\":{\"version\":1,\"train_window\":96,\"prec\":1,\"f16\":1,\"act_mx4\":0,\"act_mx8\":1,\"grad_mx8\":1,\"input_mx\":1,\"input_prec\":8,\"gn_stored\":0,\"policy\":\"\",\"optimizer\":\"muon\"},\"unwrap\":{\"version\":2,\"density_scale\":4,\"density_code\":1}}"), "save unwrap2");
    CHECK(!unet_save(u, ck1, 1, "{\"runtime\":{\"version\":1,\"train_window\":96,\"prec\":1,\"f16\":1,\"act_mx4\":0,\"act_mx8\":1,\"grad_mx8\":1,\"input_mx\":1,\"input_prec\":8,\"gn_stored\":0,\"policy\":\"\",\"optimizer\":\"muon\"},\"unwrap\":{\"version\":1,\"density_scale\":4,\"density_code\":1}}"), "save unwrap");
    unet_free(u);
    char o2[512], o1[512], o2t[512];
    snprintf(o2, sizeof o2, "%s/f2", dir); snprintf(o1, sizeof o1, "%s/f1", dir); snprintf(o2t, sizeof o2t, "%s/f2t", dir);
    CHECK(predict(argv[1], dir, ck2, o2, "") == 0, "predict unwrap2 (%s/log-f2.txt)", dir);
    CHECK(predict(argv[1], dir, ck1, o1, "") == 0, "predict unwrap (%s/log-f1.txt)", dir);
    CHECK(predict(argv[1], dir, ck2, o2t, "--tta 2") == 0, "predict unwrap2 --tta 2 (%s/log-f2t.txt)", dir);
    fit_fields_buf a, b, t;
    const int la = fit_fields_load(o2, 0, 0, 1, &a), lb = fit_fields_load(o1, 0, 0, 1, &b), lt = fit_fields_load(o2t, 0, 0, 1, &t);
    CHECK(!la && !lb && !lt, "load the fields");
    if (!la && !lb && !lt) {
        const size_t n = (size_t)a.shape[0] * a.shape[1] * a.shape[2], m = n / 2 + a.shape[2] / 2;   /* a middle cell */
        CHECK(abs(a.a[0][m] - pcode(bias[0])) <= 1 && abs(a.a[1][m] - pcode(bias[1])) <= 1, "unwrap2 phase from channels 0-1: pc %d ps %d, want %d %d", a.a[0][m], a.a[1][m], pcode(bias[0]), pcode(bias[1]));
        CHECK(abs(b.a[0][m] - pcode(bias[2])) <= 1 && abs(b.a[1][m] - pcode(bias[3])) <= 1, "unwrap phase from channels 2-3: pc %d ps %d, want %d %d", b.a[0][m], b.a[1][m], pcode(bias[2]), pcode(bias[3]));
        CHECK(a.a[10] && a.a[13] && a.f.w4_d > 0 && !a.a[14], "unwrap2: winding mod 4 planes (w4_d %d), no recto plane", (int)a.f.w4_d);
        CHECK(b.a[14] && !b.a[10], "unwrap: the recto plane, no winding mod 4 planes");
        int bad = 0; for (size_t i = 0; i < n; i++) if (t.a[0][i] != a.a[0][i] || t.a[1][i] != a.a[1][i]) bad++;
        CHECK(!bad, "--tta 2 changed the constant phase in %d cells", bad);
        /* t2 (n_y^2) of unwrap2 = channel 3 (sigmoid 1.0); under unwrap it would be channel 5 */
        CHECK(abs(a.a[3][m] - (int)lrintf(255 / (1 + expf(-bias[3])))) <= 2, "unwrap2 normals from channels 2-7: t1 %d", a.a[3][m]);
        fit_fields_free(&a); fit_fields_free(&b); fit_fields_free(&t);
    }
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_predict_unwrap2: %d FAILED (kept %s)\n", fails, dir); return 1; }
    printf("test_predict_unwrap2: ok\n");
    return 0;
}
