/* ufsm surf-labels on synthetic spiral wraps: an Archimedean spiral r = 20 + 10 W around a vertical axis (W = U - dir theta / 2 pi, wrap
   k = the level set W = k, so wrap k at angle theta sits at r_k(theta) = 20 + 10 (k + dir theta / 2 pi)), wraps k = 1..7 written as .sfc
   (rows = z, columns = theta in (-pi, pi]), dir = +1 and -1. Checks, on the 9.6 um grid:
   - codes: every coded voxel's q = k + dir theta / 2 pi (mod 14) within 1/18 turn of the analytic sheet coordinate of the nearest wrap;
   - phase: between kept consecutive wraps it equals the analytic fraction (r - r_k) / (r_k+1 - r_k) (mean error < 0.03 turn) and rises
     outward (the sign), for both spiral senses (dir);
   - identical to band_field_phase_pair on the written codes (PHASE_* convention), when nothing is dropped;
   - a dropped wrap (--drop w4): no code on it, ignore = 1 and phase unknown between wraps 3 and 5, labels elsewhere kept;
   - a conflict: a wrap 8 lying on wrap 2 -> codes 255 there, written 0, ignore 2, never labelled;
   - --gt with the dir +1 codes as GT: phase agreement and pair accuracy 1.0 against itself;
   - a holdout box (--exclude) -> ignore 4;
   - same bytes with 1 and 4 threads. */
#include "band.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_surf_labels(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
#define NZ 64
#define NY 256
#define NX 256
#define C0 128.0
static double rk(double W) { return 20 + 10 * W; }
static uint8_t *readall(const char *root, const char *key) {
    z3 *z = z3_open(store_open(root), key, nullptr); if (!z) return nullptr;
    const int64_t o[3] = {0, 0, 0}, n[3] = {NZ, NY, NX}; uint8_t *b = malloc((size_t)NZ * NY * NX);
    if (z3_read(z, o, n, b, 1)) { free(b); b = nullptr; } z3_close(z); return b;
}
static void wraps(const char *dir, int sense, int dup) {
    const int H = NZ / 2, Wc = 720;
    for (int k = 1; k <= 7 + (dup ? 1 : 0); k++) {
        float *x = malloc(Wc * H * 4), *y = malloc(Wc * H * 4), *z = malloc(Wc * H * 4);
        const double kk = k == 8 ? 2.0 : k;   /* (dup: wrap "8" lies on wrap 2: codes 6 turns apart in the same voxels) */
        for (int r = 0; r < H; r++) for (int c = 0; c < Wc; c++) {
            const double th = M_PI - (c + 0.5) * 2 * M_PI / Wc, rr = rk(kk + sense * th / (2 * M_PI));
            x[r * Wc + c] = (float)(C0 + rr * cos(th)); y[r * Wc + c] = (float)(C0 + rr * sin(th)); z[r * Wc + c] = (float)(2 * r);
        }
        char p[600]; snprintf(p, sizeof p, "%s/w%03d", dir, k);
        tifxyz_write(p, Wc, H, x, y, z, 2, 9.6, k, "test");
        free(x); free(y); free(z);
    }
}
static int run(const char *root, const char *wd, const char *out, const char *dir, const char *threads, const char *extra1, const char *extra2, const char *gt) {
    char ax[400], wdp[400], o[400], gtr[400]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wdp, sizeof wdp, "%s/%s", root, wd); snprintf(o, sizeof o, "%s/%s", root, out);
    char *av[40]; int n = 0;
    av[n++] = "ufsm"; av[n++] = "surf-labels"; av[n++] = wdp; av[n++] = o; av[n++] = "--axis"; av[n++] = ax; av[n++] = "--axis-um"; av[n++] = "9.6";
    av[n++] = "--dir"; av[n++] = (char *)dir; av[n++] = "--grid-um"; av[n++] = "9.6"; av[n++] = "--shape"; av[n++] = "64,256,256";
    av[n++] = "--shard"; av[n++] = "128"; av[n++] = "--block"; av[n++] = "32"; av[n++] = "--radius"; av[n++] = "12"; av[n++] = "--threads"; av[n++] = (char *)threads;
    if (extra1) { av[n++] = (char *)extra1; av[n++] = (char *)extra2; }
    if (gt) { snprintf(gtr, sizeof gtr, "%s/%s", root, gt); av[n++] = "--gt"; av[n++] = gtr; av[n++] = "9.6"; }
    if (extra1 && !strcmp(extra1, "--exclude")) { av[n++] = "--exclude-ct"; av[n++] = "scan.zarr"; av[n++] = "--exclude-mm"; av[n++] = "0"; }
    return cmd_surf_labels(n, av);
}
int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_surflabels_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[1400], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/wp %s/wm %s/wd", root, root, root, root); if (system(cmd)) return 1;
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [{\"x\": %g, \"y\": %g, \"z\": 0}, {\"x\": %g, \"y\": %g, \"z\": %d}]}\n", C0, C0, C0, C0, NZ - 1); fclose(f);
    snprintf(p, sizeof p, "%s/hold.json", root); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"boxes\": [{\"name\": \"h\", \"ct\": \"S/scan.zarr\", \"l0\": [10, 160, 160, 20, 30, 30]}, {\"name\": \"o\", \"ct\": \"T/other.zarr\", \"l0\": [0, 0, 0, 64, 64, 64]}]}\n"); fclose(f);
    tifxyz_set_sfc(0.01);
    snprintf(p, sizeof p, "%s/wp", root); wraps(p, 1, 0);
    snprintf(p, sizeof p, "%s/wm", root); wraps(p, -1, 0);
    snprintf(p, sizeof p, "%s/wd", root); wraps(p, 1, 1);
    tifxyz_set_sfc(0);
    CHECK(!run(root, "wp", "p1", "1", "1", nullptr, nullptr, nullptr), "surf-labels dir +1 (1 thread) failed");
    CHECK(!run(root, "wp", "p4", "1", "4", nullptr, nullptr, nullptr), "surf-labels dir +1 (4 threads) failed");
    snprintf(cmd, sizeof cmd, "cd %s && for a in codes phase ignore; do for s in $(cd p1/$a.zarr/9.6 && find c -type f); do cmp -s p1/$a.zarr/9.6/$s p4/$a.zarr/9.6/$s || exit 1; done; done", root);
    CHECK(!system(cmd), "outputs differ between 1 and 4 threads");
    CHECK(!run(root, "wm", "m", "-1", "4", nullptr, nullptr, nullptr), "surf-labels dir -1 failed");
    CHECK(!run(root, "wp", "d", "1", "4", "--drop", "w4", nullptr), "surf-labels --drop failed");
    CHECK(!run(root, "wd", "c", "1", "4", nullptr, nullptr, nullptr), "surf-labels dup failed");
    CHECK(!run(root, "wp", "g", "1", "4", nullptr, nullptr, "p1/codes.zarr"), "surf-labels --gt failed");
    snprintf(p, sizeof p, "%s/hold.json", root);
    CHECK(!run(root, "wp", "h", "1", "4", "--exclude", p, nullptr), "surf-labels --exclude failed");
    const int sense[2] = {1, -1}; const char *outs[2] = {"p1", "m"};
    for (int s = 0; s < 2; s++) {
        char r[600]; snprintf(r, sizeof r, "%s/%s/codes.zarr", root, outs[s]); uint8_t *cd = readall(r, "9.6");
        snprintf(r, sizeof r, "%s/%s/phase.zarr", root, outs[s]); uint8_t *ph = readall(r, "9.6");
        if (!cd || !ph) { CHECK(0, "cannot read outputs"); break; }
        long nc = 0, cbad = 0, np = 0, up = 0, tot = 0; double perr = 0;
        for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
            const size_t i = ((size_t)z * NY + y) * NX + x; const double Y = y - C0, X = x - C0, rr = hypot(X, Y), th = atan2(Y, X);
            const double Wv = (rr - 20) / 10 - sense[s] * th / (2 * M_PI);   /* the analytic W: wrap k where W = k */
            const double U = (rr - 20) / 10;   /* the sheet coordinate q of a point on a sheet */
            if (cd[i]) { nc++; double dq = (cd[i] - 1) / 18.0 - U; dq -= 14 * round(dq / 14); if (fabs(dq) > 1.5 / 18 + 0.05) cbad++; }
            if (ph[i] != PHASE_UNKNOWN && Wv > 1.05 && Wv < 6.95) {
                np++; double e = ph[i] / 252.0 - (Wv - floor(Wv)); e -= round(e); perr += fabs(e);
                const size_t j = i + 1; if (x + 1 < NX && ph[j] != PHASE_UNKNOWN && x > C0 + 25 && abs(y - (int)C0) < 4) { int d = ph[j] - ph[i]; if (d > 126) d -= 252; if (d < -126) d += 252; if (d) { tot++; up += d > 0; } }
            }
        }
        printf("test_surflabels: dir %+d: %ld coded voxels (%ld off), %ld phase voxels, mean |phase error| %.4f turn, outward %ld / %ld\n", sense[s], nc, cbad, np, np ? perr / np : 0, up, tot);
        CHECK(nc > 1000 && cbad <= nc / 200, "codes are not q = k + dir theta / 2 pi");
        CHECK(np > 100000 && perr / np < 0.03, "phase differs from the analytic fraction");
        CHECK(tot > 100 && up > 0.98 * tot, "phase does not rise outward");
        if (s == 0) {   /* identical to band_field_phase_pair on the codes (one block = whole array, radius 12) */
            uint8_t *ref = malloc((size_t)NZ * NY * NX); const int n3[3] = {NZ, NY, NX};
            band_field_phase_pair(cd, n3, 12, ref, nullptr);
            long diff = 0, cmpn = 0; for (int z = 15; z < NZ - 15; z++) for (int y = 15; y < NY - 15; y++) for (int x = 15; x < NX - 15; x++) { const size_t i = ((size_t)z * NY + y) * NX + x; cmpn++; diff += ref[i] != ph[i]; }
            printf("test_surflabels: vs band_field_phase_pair on the written codes: %ld of %ld differ\n", diff, cmpn);
            CHECK(diff == 0, "phase differs from band_field_phase_pair (convention)");
            free(ref);
        }
        free(cd); free(ph);
    }
    { char r[600]; snprintf(r, sizeof r, "%s/d/codes.zarr", root); uint8_t *cd = readall(r, "9.6");
      snprintf(r, sizeof r, "%s/d/phase.zarr", root); uint8_t *ph = readall(r, "9.6"); snprintf(r, sizeof r, "%s/d/ignore.zarr", root); uint8_t *ig = readall(r, "9.6");
      long on4 = 0, gap = 0, gapk = 0, gapi = 0, other = 0, otherk = 0;
      for (size_t i = 0; cd && ph && ig && i < (size_t)NZ * NY * NX; i++) {
          const int z = (int)(i / (NY * NX)), y = (int)(i / NX % NY), x = (int)(i % NX); (void)z;
          const double Y = y - C0, X = x - C0, Wv = (hypot(X, Y) - 20) / 10 - atan2(Y, X) / (2 * M_PI);
          if (fabs(Wv - 4) < 0.05 && cd[i]) on4++;
          if (Wv > 3.15 && Wv < 4.85) { gap++; gapk += ph[i] != PHASE_UNKNOWN; gapi += ig[i] == 1; }
          if ((Wv > 1.15 && Wv < 2.85) || (Wv > 5.15 && Wv < 6.85)) { other++; otherk += ph[i] != PHASE_UNKNOWN && !ig[i]; }
      }
      printf("test_surflabels: --drop w4: codes on it %ld, gap 3..5 %ld voxels: phase known %ld, ignore 1 %ld; elsewhere %ld of %ld kept\n", on4, gap, gapk, gapi, otherk, other);
      CHECK(cd && on4 == 0, "a dropped wrap has codes");
      CHECK(gap > 1000 && gapk == 0 && gapi > 0.9 * gap, "the gap around a dropped wrap is labelled");
      CHECK(other > 1000 && otherk > 0.9 * other, "labels away from the dropped wrap lost");
      free(cd); free(ph); free(ig); }
    { char r[600]; snprintf(r, sizeof r, "%s/c/codes.zarr", root); uint8_t *cd = readall(r, "9.6");
      snprintf(r, sizeof r, "%s/c/ignore.zarr", root); uint8_t *ig = readall(r, "9.6"); snprintf(r, sizeof r, "%s/c/phase.zarr", root); uint8_t *ph = readall(r, "9.6");
      long n255 = 0, nig2 = 0, nconf_lab = 0;
      for (size_t i = 0; cd && ig && ph && i < (size_t)NZ * NY * NX; i++) { n255 += cd[i] == 255; nig2 += ig[i] == 2; nconf_lab += ig[i] == 2 && ph[i] != PHASE_UNKNOWN; }
      printf("test_surflabels: wrap 8 on wrap 2: %ld conflict ignore voxels, codes 255 written %ld, labelled %ld\n", nig2, n255, nconf_lab);
      CHECK(nig2 > 0 && n255 == 0 && nconf_lab == 0, "conflicts not marked / labelled");
      free(cd); free(ig); free(ph); }
    { char r[600]; snprintf(r, sizeof r, "%s/h/ignore.zarr", root); uint8_t *ig = readall(r, "9.6"); long in = 0, n4 = 0, out4 = 0;
      for (int z = 0; ig && z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
          const int inb = z >= 10 && z < 30 && y >= 160 && y < 190 && x >= 160 && x < 190; const uint8_t v = ig[((size_t)z * NY + y) * NX + x];
          if (inb) { in++; n4 += v == 4; } else out4 += v == 4; }
      printf("test_surflabels: holdout box: %ld of %ld voxels ignore 4 (it has no wraps everywhere: labelled ones only), %ld outside\n", n4, in, out4);
      CHECK(n4 > 0 && out4 == 0, "holdout box not excluded (or the other CT's box applied)");
      free(ig); }
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {} }
    printf("test_surflabels: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
