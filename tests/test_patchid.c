/* ufsm patch-id on a synthetic spiral: sheets r = 40 + 10 W around a vertical axis (W = U - theta / 2 pi, dir +1), a thin sheet mask
   (|distance| <= 1 voxel) and GT winding codes on the sheets (q = round(W) + theta / 2 pi), wraps k = 1..7 on the sheets as .sfc, except
   that wraps 4 and 5 swap sheets for rows >= 16 (z >= 32): a local identity error. With --anchor-split-mm (GT votes only from even
   z blocks of 16 rows) and --anchor-out, the cells of the odd blocks get their offsets only through the patches (status 2):
   wrap 4 rows 24..31 D = +1, wrap 5 rows 24..31 D = -1, rows 8..15 D = 0 (both), all other wraps' odd-block cells D = 0.
   Also: the run is deterministic for 1 and 3 threads (same anchor file bytes). */
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_patch_id(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
#define NZ 64
#define NS 256
#define C0 128.0
static double rk(double W) { return 40 + 10 * W; }
static int vol(const char *root, const char *name, int codes) {   /* the sheet mask (codes 0) or the GT codes (1), one 128^3-sharded level "9.6" */
    char p[600]; snprintf(p, sizeof p, "%s/%s/9.6", root, name);
    const int64_t sh[3] = {NZ, NS, NS}; z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return 1;
    uint8_t *d = malloc(128 * 128 * 128);
    for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        memset(d, 0, 128 * 128 * 128);
        for (int z = 0; z < NZ; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
            const double Y = sy * 128 + y - C0, X = sx * 128 + x - C0, th = atan2(Y, X), Wv = (hypot(X, Y) - 40) / 10 - th / (2 * M_PI);
            const double k = round(Wv), e = (Wv - k) * 10;
            if (k < 1 || k > 7 || fabs(e) > 1) continue;
            const double q = k + th / (2 * M_PI); long c = lround(18 * q) % 252; if (c < 0) c += 252;
            d[((size_t)z * 128 + y) * 128 + x] = codes ? (uint8_t)(1 + c) : 255;
        }
        if (z3w_write_shard(w, 0, sy, sx, d, 2)) return 1;
    }
    free(d); return z3w_close(w);
}
static void wraps(const char *dir) {
    const int H = NZ / 2, Wc = 1440;
    for (int k = 1; k <= 7; k++) {
        float *x = malloc(Wc * H * 4), *y = malloc(Wc * H * 4), *z = malloc(Wc * H * 4);
        for (int r = 0; r < H; r++) for (int c = 0; c < Wc; c++) {
            const int kk = r >= 16 && (k == 4 || k == 5) ? 9 - k : k;   /* rows >= 16: wraps 4 and 5 on each other's sheet */
            const double th = M_PI - (c + 0.5) * 2 * M_PI / Wc, rr = rk(kk + th / (2 * M_PI));
            x[r * Wc + c] = (float)(C0 + rr * cos(th)); y[r * Wc + c] = (float)(C0 + rr * sin(th)); z[r * Wc + c] = (float)(2 * r);
        }
        char p[600]; snprintf(p, sizeof p, "%s/w%03d", dir, k);
        tifxyz_write(p, Wc, H, x, y, z, 2, 9.6, k, "test");
        free(x); free(y); free(z);
    }
}
static int run(const char *root, const char *out, const char *threads) {
    char ax[400], wd[400], o[400], sh[400], gt[400], an[400];
    snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wd, sizeof wd, "%s/w", root); snprintf(o, sizeof o, "%s/%s", root, out);
    snprintf(sh, sizeof sh, "%s/sheet", root); snprintf(gt, sizeof gt, "%s/gt", root); snprintf(an, sizeof an, "%s/%s.anc", root, out);
    char *av[] = {"ufsm", "patch-id", wd, o, "--axis", ax, "--axis-um", "9.6", "--dir", "1", "--sheet", sh, "9.6", "--grid-um", "9.6",
                  "--block", "16", "--band", "32", "--cells", "4,32", "--gt", gt, "9.6", "--anchor-split-mm", "0.1536", "--anchor-out", an,
                  "--min-votes", "50", "--min-anchor", "20", "--threads", (char *)threads};
    return cmd_patch_id((int)(sizeof av / sizeof *av), av);
}
int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_patchid_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[1400], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/w", root, root); if (system(cmd)) return 1;
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [{\"x\": %g, \"y\": %g, \"z\": 0}, {\"x\": %g, \"y\": %g, \"z\": %d}]}\n", C0, C0, C0, C0, NZ - 1); fclose(f);
    if (vol(root, "sheet", 0) || vol(root, "gt", 1)) { fprintf(stderr, "cannot write the volumes\n"); return 1; }
    tifxyz_set_sfc(0.01); snprintf(p, sizeof p, "%s/w", root); wraps(p); tifxyz_set_sfc(0);
    CHECK(!run(root, "a1", "1"), "patch-id (1 thread) failed");
    CHECK(!run(root, "a3", "3"), "patch-id (3 threads) failed");
    snprintf(cmd, sizeof cmd, "cmp -s %s/a1.anc %s/a3.anc", root, root); CHECK(!system(cmd), "anchor files differ between 1 and 3 threads");
    /* the anchor file: UFSMANC2, {nw, cr, ns, dir, nc}, (id, h) per wrap, D[nc], dm[nc], st[nc] */
    snprintf(p, sizeof p, "%s/a1.anc", root); f = fopen(p, "rb"); CHECK(f != nullptr, "no anchor file");
    if (f) {
        char mg[8]; int32_t hd[5]; CHECK(fread(mg, 1, 8, f) == 8 && !memcmp(mg, "UFSMANC2", 8) && fread(hd, 4, 5, f) == 5, "bad anchor header");
        const int nw = hd[0], cr = hd[1], ns = hd[2], nc = hd[4]; CHECK(nw == 7 && cr == 4 && ns == 32 && hd[3] == 1, "anchor header values");
        int32_t (*wh)[2] = malloc(sizeof *wh * nw); CHECK(fread(wh, 8, nw, f) == (size_t)nw, "wrap table");
        float *D = malloc(4 * nc), *dm = malloc(4 * nc); uint8_t *st = malloc(nc);
        CHECK(fread(D, 4, nc, f) == (size_t)nc && fread(dm, 4, nc, f) == (size_t)nc && fread(st, 1, nc, f) == (size_t)nc, "anchor body");
        int c = 0, bad = 0, n2 = 0, n2s = 0;
        for (int k = 0; k < nw; k++) { const int nrb = (wh[k][1] + cr - 1) / cr;
            for (int rb = 0; rb < nrb; rb++) for (int s = 0; s < ns; s++, c++) {
                const int row0 = rb * cr, odd = (row0 * 2 / 16) % 2;   /* (rows are z / 2; anchor blocks of 16 z) */
                if (!odd || st[c] != 2) continue;
                n2++; const int id = wh[k][0]; const double want = row0 >= 16 && id == 4 ? 1 : row0 >= 16 && id == 5 ? -1 : 0;
                if (want != 0) n2s++;
                if (!(fabs(D[c] - want) < 0.25)) bad++;
            } }
        printf("test_patchid: odd-block cells set through patches %d (%d on the swapped rows), wrong %d\n", n2, n2s, bad);
        CHECK(n2 > 400 && n2s > 100, "too few cells reached through the patches");
        CHECK(bad <= n2 / 50, "patch-propagated offsets wrong");
        fclose(f); free(wh); free(D); free(dm); free(st);
    }
    snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (!fails && system(cmd)) return 1;
    printf(fails ? "test_patchid: %d FAILED\n" : "test_patchid: ok\n", fails);
    return fails ? 1 : 0;
}
