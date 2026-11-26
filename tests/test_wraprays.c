/* wrap-rays (src/wraprays.c, PHerc1203 sheet-count check 2026-11-26) on a synthetic slice: concentric sheets about (128, 128), 16 voxels
   per sheet from r 16 to 112 (6 sheets): papyrus 120 with two bright fibre bands (200, offsets 3 and 9, 2 wide) and a 4-voxel air gap (20)
   at offsets 12..15; outside 0. Six cylinder meshes at the sheets' middles (r 16 k + 22). Per ray: 12 laminae (two bands per sheet), 6
   air-separated (one per sheet), 6 mesh crossings; the window's air share ~0.22 (5 inner gaps over the object span). CPU only (links the fit model code). */
#include "z3w.h"
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_rays(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

int main(void) {
    const char *t = getenv("TMPDIR"); char dir[600], vol[700], md[700], p[900], out[700];
    snprintf(dir, sizeof dir, "%s/wraprays.%d", t && *t ? t : "/tmp", (int)getpid());
    snprintf(vol, sizeof vol, "%s/vol", dir); snprintf(md, sizeof md, "%s/meshes", dir); snprintf(out, sizeof out, "%s/r.tsv", dir);
    snprintf(p, sizeof p, "mkdir -p %s", md); if (system(p)) return 1;
    const int64_t shape[3] = {256, 256, 256}; uint8_t *d = calloc((size_t)256 * 256 * 256, 1), *sl = malloc(256 * 256);
    for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) {
        const double r = hypot(y - 128.0, x - 128.0); uint8_t v = 0;
        if (r >= 16 && r < 112) { const double o = fmod(r - 16, 16); v = o >= 12 ? 20 : (o >= 3 && o < 5) || (o >= 9 && o < 11) ? 200 : 120; }
        sl[y * 256 + x] = v;
    }
    for (int z = 0; z < 256; z++) memcpy(d + (size_t)z * 256 * 256, sl, 256 * 256);
    z3w *w = z3w_create(vol, shape, 256, 0.f, 0, nullptr); CHECK(w != nullptr, "create %s", vol); if (!w) return 1;
    CHECK(!z3w_write_shard(w, 0, 0, 0, d, 2), "write shard"); z3w_close(w);
    const int R = 57, C = 720; float *X = malloc(sizeof *X * R * C), *Y = malloc(sizeof *Y * R * C), *Z = malloc(sizeof *Z * R * C);
    for (int k = 0; k < 6; k++) {
        const double rad = 16 * k + 22;
        for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) {
            const double a = 2 * M_PI * (c + 0.25) / (C - 1);   /* the last column overlaps the first: a closed ring */
            X[r * C + c] = (float)(128 + rad * cos(a)); Y[r * C + c] = (float)(128 + rad * sin(a)); Z[r * C + c] = 100.5f + r;
        }
        char m[800]; snprintf(m, sizeof m, "%s/w%d", md, k);
        CHECK(!tifxyz_write(m, C, R, X, Y, Z, 1.0, 1.0, -1, "test_wraprays"), "write mesh %d", k);
    }
    char *av[] = {"ufsm", "wrap-rays", out, "--ct", vol, ".", "--level", "0", "--um", "1", "--z", "128", "--center", "128,128", "--meshes", md,
                  "--rays", "8", "--theta0", "10", "--win-um", "120", "--prom", "12", "--sep", "3", nullptr};
    CHECK(cmd_wrap_rays(26, av) == 0, "wrap-rays");
    FILE *f = fopen(out, "r"); char line[512]; int nr = 0, first = 1;
    while (f && fgets(line, sizeof line, f)) {
        if (first) { first = 0; continue; }
        double z, th, r0, r1, cf, air; int lam, sh, wr; char cls[16];
        if (sscanf(line, "%lf %lf %lf %lf %d %d %d %lf %lf %15s", &z, &th, &r0, &r1, &lam, &sh, &wr, &cf, &air, cls) != 10) { CHECK(0, "bad row %s", line); continue; }
        nr++;
        CHECK(lam == 12, "theta %.0f: %d laminae, want 12", th, lam);
        CHECK(sh == 6, "theta %.0f: %d air-separated laminae, want 6", th, sh);
        CHECK(wr == 6, "theta %.0f: %d mesh crossings, want 6", th, wr);
        CHECK(air > 0.15 && air < 0.30 && strcmp(cls, "packed"), "theta %.0f: air %.3f class %s, want 5 gaps of 4 in ~92 (0.22), not packed", th, air, cls);
    }
    if (f) fclose(f);
    CHECK(nr == 8, "%d windows, want 8 (one per ray)", nr);
    free(X); free(Y); free(Z); free(d); free(sl);
    snprintf(p, sizeof p, "rm -rf -- '%s'", dir); if (strstr(dir, "/wraprays.")) (void)!system(p);
    printf("%s: test_wraprays (%d windows)\n", fails ? "FAIL" : "PASS", nr);
    return fails ? 1 : 0;
}
