/* predict --feather B (one-sided sin^2 blend of each window border from the earlier windows' upper halo predictions). A small random
   2-level net (GroupNorm: per-window statistics, so neighbouring windows disagree in their overlap) on a 256^3 CT fixture, window 160
   halo 16 shard 128 (one window per shard):
   - away from the lower shells (every coordinate mod 128 >= B) the feathered output equals the plain predict byte for byte, and on the
     box's first window (no earlier windows) everywhere;
   - in a lower z shell the output is the sin^2 blend of the window's own and the lower window's prediction (that window run alone);
   - --resume after a run of the lower half (the done windows recomputed for their spills only) and --gpus 0,0 (two workers, contiguous
     window layers, each computing the layer before its own unwritten) give the same bytes as the single run.
   Usage: test_predict_feather <ufsm binary> */
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
enum { N = 256, SH = 128, B = 16 };

static int predict(const char *ufsm, const char *dir, const char *name, const char *extra) {   /* extra: options, or a whole geometry (starts with --window) */
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "%s predict %s/r.ckpt %s ct %s/%s --um 1 --level 0 %s --ema 1 --threads 2 --levels 1 --q 0 "
             "%s >> %s/log-%s.txt 2>&1", ufsm, dir, dir, dir, name, strncmp(extra, "--window", 8) ? "--window 160 --halo 16 --shard 128" : "", extra, dir, name);
    return system(cmd);
}
static uint8_t *load(const char *dir, const char *name, int M) {   /* the first M^3 voxels */
    char p[600]; snprintf(p, sizeof p, "%s/%s", dir, name);
    store *s = store_open(p); z3 *z = s ? z3_open(s, "1", nullptr) : nullptr;
    uint8_t *v = malloc((size_t)M * M * M); const int64_t o[3] = {0, 0, 0}, n[3] = {M, M, M};
    if (!z || !v || z3_read(z, o, n, v, 2)) { fprintf(stderr, "cannot read %s: %s\n", p, z3_error()); free(v); v = nullptr; }
    if (z) z3_close(z);
    if (s) store_close(s);
    return v;
}
static double jump(const uint8_t *v, int b) {   /* mean |dv| across the z / y / x planes at b, minus the same at the planes b +- 8.. (interior) */
    double sb = 0, si = 0; size_t nb = 0, ni = 0;
    for (int k = 1; k < N; k++) {
        const int border = k == b;
        if (!border && (k % SH) < B + 2) continue;   /* (the feathered shell planes count as neither) */
        for (int u = 0; u < N; u++) for (int w = 0; w < N; w++) {
            const size_t i1 = ((size_t)k * N + u) * N + w, i0 = i1 - (size_t)N * N;
            const size_t j1 = ((size_t)u * N + k) * N + w, j0 = j1 - N;
            const size_t l1 = ((size_t)u * N + w) * N + k, l0 = l1 - 1;
            const double d = abs(v[i1] - v[i0]) + abs(v[j1] - v[j0]) + abs(v[l1] - v[l0]);
            if (border) { sb += d; nb += 3; } else { si += d; ni += 3; }
        }
    }
    return sb / nb - si / ni;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test_predict_feather <ufsm binary>\n"); return 2; }
    char dir[] = "/tmp/test_pfe_XXXXXX"; if (!mkdtemp(dir)) return 1;   /* (a 256^3 CT, outputs ~16 MB each) */
    { const int64_t sh[3] = {N, N, N}; const double um = 1; char p[512]; uint8_t *d = malloc((size_t)N * N * N);
      for (size_t i = 0; i < (size_t)N * N * N; i++) {   /* structure + noise: material everywhere */
          const int z = (int)(i / (N * N)), y = (int)(i / N % N), x = (int)(i % N);
          d[i] = (uint8_t)(20 + 0.5 * z + 0.2 * y + 30 * sin(0.11 * x + 0.07 * y) * cos(0.05 * z) * (1 + (x > 140)) + (int)((i * 2654435761u) >> 28));   /* (ramps + a contrast step: per-window statistics differ) */
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
    if (nn_init(0)) { fprintf(stderr, "test_predict_feather: no GPU\n"); return 1; }
    unet_cfg cfg = {0}; cfg.nlev = 2; cfg.widths[0] = 32; cfg.widths[1] = 64; cfg.cin = 4; cfg.cout = 1; cfg.G = 8; cfg.down_norm = 1;
    unet *u = unet_create(&cfg); unet_init(u, 11); unet_ema(u, 0.f);
    char ck[512]; snprintf(ck, sizeof ck, "%s/r.ckpt", dir);
    CHECK(!unet_save(u, ck, 1, "{\"runtime\":{\"version\":1,\"train_window\":96,\"prec\":1,\"f16\":1,\"act_mx4\":0,\"act_mx8\":1,\"grad_mx8\":1,\"input_mx\":1,\"input_prec\":8,\"gn_stored\":0,\"policy\":\"\",\"optimizer\":\"muon\"}}"), "save");
    unet_free(u);
    CHECK(predict(argv[1], dir, "plain", "") == 0, "plain predict (%s/log-plain.txt)", dir);
    CHECK(predict(argv[1], dir, "fe", "--feather 16") == 0, "--feather (%s/log-fe.txt)", dir);
    CHECK(predict(argv[1], dir, "fer", "--feather 16 --full-grid 1 --resume 1 --box 0,0,0,128,256,256") == 0 &&
          predict(argv[1], dir, "fer", "--feather 16 --full-grid 1 --resume 1 --box 0,0,0,256,256,256") == 0, "--feather --resume (%s/log-fer.txt)", dir);
    CHECK(predict(argv[1], dir, "feg", "--feather 16 --gpus 0,0 --box 0,0,0,256,256,256") == 0, "--feather --gpus 0,0 (%s/log-feg.txt)", dir);
    /* the blend weights: a box at 32 (windows [16, 176), [144, 304) per axis) feathered and plain, and the first of those windows alone
       (window 160 halo 0 at 16, one shard: one window): at z = 160 + j (j < B; y, x inside the first window's core) feathered = u(j) own + (1 - u(j)) lower window */
    CHECK(predict(argv[1], dir, "fe2", "--feather 16 --box 32,32,32,224,224,224") == 0 && predict(argv[1], dir, "pl2", "--box 32,32,32,224,224,224") == 0 &&
          predict(argv[1], dir, "w0", "--window 160 --halo 0 --shard 256 --box 16,16,16,160,160,160") == 0, "blend-weight runs (%s/log-fe2.txt, log-pl2.txt, log-w0.txt)", dir);
    { uint8_t *f2 = load(dir, "fe2", 224), *p2 = load(dir, "pl2", 224), *w0 = load(dir, "w0", 160); size_t nck = 0, bad = 0, moved = 0;
      if (f2 && p2 && w0) for (int j = 0; j < B; j++) for (int y = 48; y < 144; y += 3) for (int x = 48; x < 144; x += 3) {
          const double sj = sin(M_PI / 2 * (j + 0.5) / B), uj = sj * sj;
          const size_t k2 = ((size_t)(128 + j) * 224 + (y - 32)) * 224 + (x - 32), k0 = ((size_t)(144 + j) * 160 + (y - 16)) * 160 + (x - 16);
          const double want = uj * p2[k2] + (1 - uj) * w0[k0];
          nck++; bad += fabs(f2[k2] - want) > 1.0; moved += f2[k2] != p2[k2];
      } else CHECK(0, "blend-weight outputs");
      CHECK(nck && !bad, "%zu of %zu z-shell voxels off the sin^2 blend of the two windows", bad, nck);
      fprintf(stderr, "test_predict_feather: %zu z-shell voxels checked, %zu differ from the plain predict\n", nck, moved);
      free(f2); free(p2); free(w0); }
    uint8_t *a = load(dir, "plain", N), *f = load(dir, "fe", N), *r = load(dir, "fer", N), *g = load(dir, "feg", N);
    if (a && f && r && g) {
        size_t core = 0, cdiff = 0, first = 0, fdiff = 0, shell = 0, sdiff = 0, rdiff = 0, gdiff = 0;
        for (size_t i = 0; i < (size_t)N * N * N; i++) {
            const int z = (int)(i / (N * N)), y = (int)(i / N % N), x = (int)(i % N);
            const int insh = z % SH < B || y % SH < B || x % SH < B, win0 = z < SH && y < SH && x < SH;
            if (!insh) { core++; cdiff += a[i] != f[i]; }
            else if (win0) { first++; fdiff += a[i] != f[i]; }
            else { shell++; sdiff += a[i] != f[i]; }
            rdiff += r[i] != f[i]; gdiff += g[i] != f[i];
        }
        CHECK(!cdiff, "%zu of %zu core voxels differ from the plain predict", cdiff, core);
        CHECK(!fdiff, "%zu of %zu lower-shell voxels of the first window differ from the plain predict", fdiff, first);
        CHECK(sdiff > shell / 4, "only %zu of %zu blended shell voxels changed (the windows should disagree)", sdiff, shell);
        CHECK(!rdiff, "--resume: %zu voxels differ from the single run", rdiff);
        CHECK(!gdiff, "--gpus 0,0: %zu voxels differ from the single run", gdiff);
        const double ja = jump(a, SH), jf = jump(f, SH);
        fprintf(stderr, "test_predict_feather: border jump plain %.3f, feathered %.3f; %zu / %zu shell voxels blended\n", ja, jf, sdiff, shell);
        CHECK(jf <= ja + 0.1, "border jump plain %.3f feathered %.3f", ja, jf);
    } else CHECK(0, "outputs");
    free(a); free(f); free(r); free(g);
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_predict_feather: %d FAILED (kept %s)\n", fails, dir); return 1; }
    printf("test_predict_feather: ok\n");
    return 0;
}
