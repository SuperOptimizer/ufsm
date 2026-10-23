/* ufsm reg render on synthetic data: a base scan (1 unit voxels), an identical copy (identity map) and a half-resolution scan (2 unit
   voxels, affine c = 0.5 f + 1.25, reached f2c), written as volcomp zarr pyramids; a planar tifxyz through them. The copy's baseline
   difference must vanish, the base must be fully covered, and a failing evaluation box of the half scan's pair must blank its disk
   (and only it). */
#include "reg.h"
#include "json.h"
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int cmd_reg_render(int argc, char **argv);
int cmd_reg_inkstat(int argc, char **argv);
int cmd_sr_pairs(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double tex(const double P[3]) {
    return 120 + 40 * sin(P[0] * 0.31 + 0.2 * P[2]) * cos(P[1] * 0.23 - 0.1 * P[2]) + 25 * sin(0.17 * (P[0] + P[1]) + 0.29 * P[2]);
}
/* one OME pyramid: level 0 n^3 at voxel um (value at voxel c = mean of tex over its cube, physical P = (c - t) / s), 3 levels */
static void write_vol(const char *dir, int n, double um, double s, double t) {
    mkdir(dir, 0755);
    uint8_t *v = calloc((size_t)128 * 128 * 128, 1);
    for (int z = 0; z < n; z++) for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
        double a = 0; const int k = s < 1 ? 2 : 1;
        for (int q = 0; q < k * k * k; q++) {
            const double c[3] = {x + (k > 1 ? ((q & 1) - 0.5) * 0.5 : 0), y + (k > 1 ? (((q >> 1) & 1) - 0.5) * 0.5 : 0), z + (k > 1 ? ((q >> 2) - 0.5) * 0.5 : 0)};
            const double P[3] = {(c[0] - t) / s, (c[1] - t) / s, (c[2] - t) / s}; a += tex(P);
        }
        v[((size_t)z * 128 + y) * 128 + x] = (uint8_t)fmin(255, fmax(1, a / (k * k * k) + 0.5));
    }
    double ums[3]; int m = n;
    for (int l = 0; l < 3; l++) {
        ums[l] = um * (1 << l);
        char lv[64], p[1200]; z3w_level_name(ums[l], lv, sizeof lv); snprintf(p, sizeof p, "%s/%s", dir, lv);
        const int64_t sh[3] = {m, m, m};
        z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); z3w_write_shard(w, 0, 0, 0, v, 1); z3w_close(w);
        const int m2 = (m + 1) / 2;   /* mean pool in place into the 128^3 buffer */
        uint8_t *o = calloc((size_t)128 * 128 * 128, 1);
        for (int z = 0; z < m2; z++) for (int y = 0; y < m2; y++) for (int x = 0; x < m2; x++) {
            int sum = 0, c = 0; for (int q = 0; q < 8; q++) { const int zz = 2 * z + (q >> 2), yy = 2 * y + (q >> 1 & 1), xx = 2 * x + (q & 1); if (zz < m && yy < m && xx < m) { sum += v[((size_t)zz * 128 + yy) * 128 + xx]; c++; } }
            o[((size_t)z * 128 + y) * 128 + x] = (uint8_t)((sum + c / 2) / c);
        }
        free(v); v = o; m = m2;
    }
    z3w_write_group(dir, ums, 3, "test", nullptr);
    free(v);
}
static void write_map(const char *pairdir, double s, double t, double fum, const double *bad) {
    char p[1200]; mkdir(pairdir, 0755); snprintf(p, sizeof p, "%s/final", pairdir); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/final/reg.json", pairdir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"coarse\":{\"um\":%g},\"fine\":{\"um\":1},\"A\":[[%g,0,0,%g],[0,%g,0,%g],[0,0,%g,%g]]}\n", fum, s, t, s, t, s, t); fclose(f);
    snprintf(p, sizeof p, "%s/final/residuals.jsonl", pairdir); f = fopen(p, "w");
    for (int z = 0; z <= 128; z += 4) for (int y = 0; y <= 128; y += 4) for (int x = 0; x <= 128; x += 4)
        fprintf(f, "{\"c\":[%d,%d,%d],\"v\":[0,0,0],\"pred\":[0.1,0,0],\"used\":1}\n", x, y, z);
    fclose(f);
    if (bad) { snprintf(p, sizeof p, "%s/eval_after.jsonl", pairdir); f = fopen(p, "w");
        fprintf(f, "{\"c\":[%g,%g,%g],\"s\":[1.5,0,0],\"ncc\":0.9,\"status\":\"ok\"}\n{\"c\":[60,60,60],\"s\":[0.05,0,0],\"ncc\":0.9,\"status\":\"ok\"}\n", bad[0], bad[1], bad[2]); fclose(f); }
}

int main(void) {
    char root[] = "/tmp/test_reg_render_XXXXXX"; if (!mkdtemp(root)) return 1;
    char p[1200], q[1200];
    snprintf(p, sizeof p, "%s/base.zarr", root); write_vol(p, 128, 1.0, 1.0, 0);
    snprintf(p, sizeof p, "%s/copy.zarr", root); write_vol(p, 128, 1.0, 1.0, 0);
    snprintf(p, sizeof p, "%s/half.zarr", root); write_vol(p, 66, 2.0, 0.5, 1.25);
    snprintf(p, sizeof p, "%s/pair_copy", root); write_map(p, 1.0, 0, 1.0, nullptr);
    const double bad[3] = {0.5 * 40 + 1.25, 0.5 * 40 + 1.25, 0.5 * 64 + 1.25};   /* coarse coords of base (40, 40, 64) */
    snprintf(p, sizeof p, "%s/pair_half", root); write_map(p, 0.5, 1.25, 2.0, bad);
    /* planar tifxyz z = 64, x, y 16 .. 112, step 4 */
    const int g = 25; float X[625], Y[625], Z[625];
    for (int j = 0; j < g; j++) for (int i = 0; i < g; i++) { X[j * g + i] = 16 + 4 * i; Y[j * g + i] = 16 + 4 * j; Z[j * g + i] = 64; }
    snprintf(p, sizeof p, "%s/seg", root); tifxyz_write(p, g, g, X, Y, Z, 4, 1.0, -1, "test");
    /* labels on the render grid (48 x 48 half-res px; px (x, y) = base (16 + 2x + 0.5, 16 + 2y + 0.5, 64)): ink where the texture is
       high, supervised everywhere except a 4-px frame */
    { uint8_t ink[48 * 48], sup[48 * 48];
      for (int y = 0; y < 48; y++) for (int x = 0; x < 48; x++) { const double P[3] = {16 + 2 * x + 0.5, 16 + 2 * y + 0.5, 64}; ink[y * 48 + x] = tex(P) > 125 ? 255 : 0; sup[y * 48 + x] = x >= 4 && y >= 4 && x < 44 && y < 44 ? 255 : 0; }
      snprintf(q, sizeof q, "%s/ink.tif", root); tiffw_write(q, 48, 48, 1, ink, 0); snprintf(q, sizeof q, "%s/sup.tif", root); tiffw_write(q, 48, 48, 1, sup, 0); }
    snprintf(p, sizeof p, "%s/cfg.json", root); FILE *f = fopen(p, "w");
    fprintf(f, "{\"tifxyz\":\"%s/seg\",\"downscale\":1,\"out\":\"%s/out\",\"layers\":1,\"match_um\":2,\"cover_mm\":0.012,\"tol_um\":0.5,\"threads\":2,\"cache_mb\":64,\"depth\":[-3,3],\"labels\":{\"ink\":\"%s/ink.tif\",\"sup\":\"%s/sup.tif\"},\n"
               " \"scans\":[{\"name\":\"base\",\"root\":\"%s\",\"key\":\"base.zarr\",\"um\":1,\"chain\":[]},\n"
               "  {\"name\":\"copy\",\"root\":\"%s\",\"key\":\"copy.zarr\",\"um\":1,\"chain\":[{\"map\":\"%s/pair_copy/final/reg.json\",\"dir\":\"c2f\",\"pair\":\"%s/pair_copy\"}]},\n"
               "  {\"name\":\"half\",\"root\":\"%s\",\"key\":\"half.zarr\",\"um\":2,\"chain\":[{\"map\":\"%s/pair_half/final/reg.json\",\"dir\":\"f2c\",\"pair\":\"%s/pair_half\"}]}],\n"
               " \"rgb\":[\"copy\",\"base\",\"half\"],\"diff\":[[\"base\",\"half\"],[\"copy\",\"half\"],[\"base\",\"copy\"]],\"baseline\":[\"base\",\"copy\"]}\n",
            root, root, root, root, root, root, root, root, root, root, root);
    fclose(f);
    char *av[] = {"ufsm", "reg", "render", p};
    CHECK(cmd_reg_render(4, av) == 0, "render failed");
    snprintf(q, sizeof q, "%s/out/stats.json", root); f = fopen(q, "r"); char buf[1 << 16]; size_t n = f ? fread(buf, 1, sizeof buf - 1, f) : 0; if (f) fclose(f); buf[n] = 0;
    json *s = json_parse(buf, n); CHECK(s != nullptr, "no stats.json");
    if (s) {
        const json *nat = json_path(s, "variants.native");
        const double cb = json_num(json_path(nat, "scans.base.coverage"), 0), cc = json_num(json_path(nat, "scans.copy.coverage"), 0), ch = json_num(json_path(nat, "scans.half.coverage"), 0);
        const json *d = json_get(nat, "diffs"); double bp99 = 1, hp99 = 0;
        for (size_t k = 0; d && k < d->n; k++) { const json *e = json_at(d, k); if (json_num(json_get(e, "baseline"), 0) > 0) bp99 = json_num(json_get(e, "p99_abs"), 1); else if (!strcmp(json_str(json_get(e, "b"), ""), "half")) hp99 = fmax(hp99, json_num(json_get(e, "p99_abs"), 0)); }
        printf("coverage base %.3f copy %.3f half %.3f; |base - copy| p99 %.4f, |base - half| p99 %.4f\n", cb, cc, ch, bp99, hp99);
        CHECK(cb > 0.8 && fabs(cb - cc) < 1e-6, "base / copy coverage %.3f %.3f", cb, cc);
        /* (the mesh border grid steps are not rendered: ~84% of the frame) the failing box blanks a disk of radius 3 coarse = 6 base voxels: ~1.3% */
        CHECK(fabs(ch - cb) < 0.005, "images are not masked by certification: half coverage %.3f vs base %.3f", ch, cb);
        CHECK(bp99 < 0.01, "identical scans differ: p99 %.4f", bp99);
        CHECK(hp99 < 0.35, "base vs half differ too much: p99 %.4f", hp99);
        json_free(s);
    }
    /* inkstat over the dir twice (two "segments": leave-one-out works) */
    snprintf(q, sizeof q, "%s/out", root); char o2[1200]; snprintf(o2, sizeof o2, "%s/ink.json", root);
    { char *ia[] = {"ufsm", "reg", "inkstat", q, q, "-o", o2}; CHECK(cmd_reg_inkstat(7, ia) == 0, "inkstat failed"); }
    { FILE *g = fopen(o2, "r"); char b2[1 << 17]; size_t m = g ? fread(b2, 1, sizeof b2 - 1, g) : 0; if (g) fclose(g); b2[m] = 0; json *ij = json_parse(b2, m);
      CHECK(ij != nullptr, "no ink.json");
      double ab = 0, al = 0; const json *ch = json_path(ij, "segments.0.channels");
      for (size_t k = 0; ch && k < ch->n; k++) { const json *e = json_at(ch, k); if (!strcmp(json_str(json_get(e, "channel"), ""), "base") && !strcmp(json_str(json_get(e, "variant"), ""), "native")) ab = json_num(json_get(e, "sheet_auc"), 0); }
      const json *lo = json_get(ij, "loso"); if (lo && lo->n) al = json_num(json_get(json_at(lo, 0), "lda3_auc"), 0);
      /* certification: the failing box's disk is dropped from the half scan's samples, kept with --all */
      double nh = 0, nh_all = 0; for (size_t k = 0; ch && k < ch->n; k++) { const json *e = json_at(ch, k); if (!strcmp(json_str(json_get(e, "channel"), ""), "half") && !strcmp(json_str(json_get(e, "variant"), ""), "native")) nh = json_num(json_get(e, "n_ink"), 0) + json_num(json_get(e, "n_noink"), 0); }
      char o3[1200]; snprintf(o3, sizeof o3, "%s/ink_all.json", root); char *ia2[] = {"ufsm", "reg", "inkstat", q, "--all", "-o", o3}; cmd_reg_inkstat(7, ia2);
      { FILE *g2 = fopen(o3, "r"); char b3[1 << 17]; size_t m3 = g2 ? fread(b3, 1, sizeof b3 - 1, g2) : 0; if (g2) fclose(g2); b3[m3] = 0; json *aj = json_parse(b3, m3); const json *c2 = json_path(aj, "segments.0.channels");
        for (size_t k = 0; c2 && k < c2->n; k++) { const json *e = json_at(c2, k); if (!strcmp(json_str(json_get(e, "channel"), ""), "half") && !strcmp(json_str(json_get(e, "variant"), ""), "native")) nh_all = json_num(json_get(e, "n_ink"), 0) + json_num(json_get(e, "n_noink"), 0); }
        json_free(aj); }
      printf("half samples: certified %.0f, all %.0f\n", nh, nh_all);
      CHECK(nh_all > nh + 10 && nh > 0.9 * nh_all, "certified %.0f vs all %.0f", nh, nh_all);
      printf("ink: base channel sheet AUC %.3f, LOSO LDA-3 AUC %.3f\n", ab, al);
      CHECK(ab > 0.85, "labels follow the base texture: AUC %.3f", ab);
      CHECK(al > 0.8, "LDA on the energies: AUC %.3f", al);
      json_free(ij); }
    const char *fs[] = {"native_base.jpg", "matched_half.jpg", "native_rgb.jpg", "native_diff.jpg", "matched_baseline.jpg", "coverage.jpg", "native_rgb_ink.jpg", "samples.bin"};
    for (int k = 0; k < 8; k++) { snprintf(q, sizeof q, "%s/out/%s", root, fs[k]); struct stat sb; CHECK(stat(q, &sb) == 0 && sb.st_size > 100, "missing %s", fs[k]); }
    /* sr-pairs: the half scan (2 units) as the input, the base scan as the target on the input's x4 grid */
    {   char pd[1200], cmd[4000]; snprintf(pd, sizeof pd, "%s/pair_sr", root);
        snprintf(cmd, sizeof cmd, "mkdir -p %s/final && cp %s/pair_half/final/residuals.jsonl %s/final/ && cp %s/pair_half/eval_after.jsonl %s/", pd, root, pd, root, pd); int r = system(cmd); (void)r;
        snprintf(p, sizeof p, "%s/final/reg.json", pd); f = fopen(p, "w");
        fprintf(f, "{\"coarse\":{\"root\":\"%s\",\"key\":\"half.zarr\",\"um\":2},\"fine\":{\"root\":\"%s\",\"key\":\"base.zarr\",\"um\":1},\"A\":[[0.5,0,0,1.25],[0,0.5,0,1.25],[0,0,0.5,1.25]]}\n", root, root); fclose(f);
        char od[1200]; snprintf(od, sizeof od, "%s/sr", root);
        char *sa[] = {"ufsm", "sr-pairs", pd, od, "--n", "6", "--in", "8", "--spacing", "6", "--threads", "2", "--cover-mm", "0.012", "--tol-um", "1", "--minfrac", "0.95"};
        CHECK(cmd_sr_pairs(18, sa) == 0, "sr-pairs failed");
        snprintf(q, sizeof q, "%s/boxes.json", od); FILE *g = fopen(q, "r"); char b4[1 << 16]; size_t m4 = g ? fread(b4, 1, sizeof b4 - 1, g) : 0; if (g) fclose(g); b4[m4] = 0;
        json *bj = json_parse(b4, m4); const int nb = (int)json_num(json_get(bj, "n"), 0); json_free(bj);
        snprintf(q, sizeof q, "%s/intensity_map.json", od); g = fopen(q, "r"); m4 = g ? fread(b4, 1, sizeof b4 - 1, g) : 0; if (g) fclose(g); b4[m4] = 0;
        json *ij2 = json_parse(b4, m4); const double r2 = json_num(json_get(ij2, "pearson_r"), 0); json_free(ij2);
        printf("sr-pairs: %d boxes, target(x1/4) vs input r %.3f\n", nb, r2);
        CHECK(nb == 6, "sr-pairs boxes %d", nb);
        CHECK(r2 > 0.9, "the x4 target pooled back must match the input: r %.3f", r2);
    }
    snprintf(q, sizeof q, "rm -rf %s", root); if (!fails) { int r = system(q); (void)r; }
    printf(fails ? "test_reg_render: %d FAILED\n" : "test_reg_render: all passed\n", fails);
    return fails != 0;
}
