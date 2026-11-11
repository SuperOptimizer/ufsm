/* ufsm fiber-trace on a synthetic prediction (raw zarr v2 arrays presence / nx / ny, 64 x 128 x 128 at scaledown 3, chunks 64): five
   straight fibres along x, three along z and a quarter circle of radius 40 in a z plane, each a tube of radius 1.5 voxels with the
   published encoding (axis on the z >= 0 hemisphere, (nx, ny) = 128 + 127 (x, y)). Checks: exactly one streamline per fibre (a seed on
   an earlier line is skipped), every streamline point within 0.75 voxel of its fibre's axis, the length within 6 voxels of the fibre's,
   family H for the x fibres and the arc, V for the z fibres, and the UFIB points are level-0 voxels ((p + 0.5) 8 - 0.5). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"

int cmd_fiber_trace(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
enum { NZ = 64, NY = 128, NX = 128, NF = 9 };
/* fibre k: the nearest axis point to (z, y, x) -> its distance; t = the unit tangent (z, y, x) there; inside = within the fibre's extent */
static double fibre(int k, double z, double y, double x, double t[3], int *inside) {
    if (k < 5) { const double fz = 10 + 10 * k, fy = 16 + 22 * k; t[0] = 0; t[1] = 0; t[2] = 1; *inside = x >= 10 && x <= 117; return hypot(z - fz, y - fy); }
    if (k < 8) { const double fy = 27 + 22 * (k - 5), fx = 25 + 35 * (k - 5) + 4; t[0] = 1; t[1] = 0; t[2] = 0; *inside = z >= 4 && z <= 59; return hypot(y - fy, x - fx); }
    const double cy = 120, cx = 120, r = hypot(y - cy, x - cx), a = atan2(y - cy, x - cx);   /* the arc: centre (120, 120), radius 40, the quadrant y, x < centre, z = 57 */
    t[0] = 0; t[1] = cos(a); t[2] = -sin(a); *inside = y < cy - 4 && x < cx - 4; return hypot(z - 57, r - 40);
}
static double fibre_len(int k) { return k < 5 ? 107 : k < 8 ? 55 : 40 * (M_PI / 2 - 2 * asin(4.0 / 40)); }

int main(void) {
    char root[] = "/tmp/ufsm-fibertrace-XXXXXX"; if (!mkdtemp(root)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    const size_t nv = (size_t)NZ * NY * NX; uint8_t *v[3]; for (int c = 0; c < 3; c++) { v[c] = malloc(nv); memset(v[c], c ? 128 : 0, nv); }
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) for (int k = 0; k < NF; k++) {
        double t[3]; int in; const double d = fibre(k, z, y, x, t, &in); if (!in || d > 1.5) continue;
        const size_t i = ((size_t)z * NY + y) * NX + x; if (t[0] < 0) { t[0] = -t[0]; t[1] = -t[1]; t[2] = -t[2]; }
        v[0][i] = (uint8_t)lround(255 - 60 * d); v[1][i] = (uint8_t)lround(128 + 127 * t[2]); v[2][i] = (uint8_t)lround(128 + 127 * t[1]);
    }
    static const char *const nm[3] = {"presence", "nx", "ny"}; char p[900];
    for (int c = 0; c < 3; c++) {
        snprintf(p, sizeof p, "mkdir -p %s/pred_%s.ome.zarr/3/0/0 %s/pred_%s.ome.zarr/3/0/1", root, nm[c], root, nm[c]); if (system(p)) return 1;
        snprintf(p, sizeof p, "%s/pred_%s.ome.zarr/3/.zarray", root, nm[c]); FILE *f = fopen(p, "w"); if (!f) return 1;
        fprintf(f, "{\"shape\": [%d, %d, %d], \"chunks\": [64, 64, 64], \"dtype\": \"|u1\", \"fill_value\": 0, \"order\": \"C\", \"filters\": null, \"dimension_separator\": \"/\", \"compressor\": null, \"zarr_format\": 2}\n", NZ, NY, NX); fclose(f);
        for (int cy = 0; cy < 2; cy++) for (int cx = 0; cx < 2; cx++) {
            snprintf(p, sizeof p, "%s/pred_%s.ome.zarr/3/0/%d/%d", root, nm[c], cy, cx); f = fopen(p, "wb"); if (!f) return 1;
            for (int z = 0; z < 64; z++) for (int y = 0; y < 64; y++) fwrite(v[c] + ((size_t)z * NY + cy * 64 + y) * NX + cx * 64, 1, 64, f);
            fclose(f);
        }
    }
    char out[400], key[400]; snprintf(out, sizeof out, "%s/t.fib", root); snprintf(key, sizeof key, "pred");
    char *av[] = {"ufsm", "fiber-trace", root, key, out, "--box", "0,0,0,64,128,128", "--scaledown", "3", "--threads", "2"};
    CHECK(!cmd_fiber_trace(11, av), "fiber-trace");
    fib_set F = {0}; CHECK(!fib_load(out, &F), "load the streamlines");
    CHECK(F.n == NF, "%d streamlines, want %d (one per fibre)", F.n, NF);
    int seen[NF] = {0};
    for (int r = 0; r < F.n; r++) {
        const int a = F.off[r], b = F.off[r + 1]; const float *m = F.pts + 3 * (size_t)((a + b) / 2); double t[3], best = 1e9; int k = -1, in;
        for (int j = 0; j < NF; j++) { const double d = fibre(j, (m[0] + 0.5) / 8 - 0.5, (m[1] + 0.5) / 8 - 0.5, (m[2] + 0.5) / 8 - 0.5, t, &in); if (d < best) { best = d; k = j; } }
        seen[k]++; double worst = 0, len = 0;
        for (int i = a; i < b; i++) { const float *q = F.pts + 3 * (size_t)i; const double d = fibre(k, (q[0] + 0.5) / 8 - 0.5, (q[1] + 0.5) / 8 - 0.5, (q[2] + 0.5) / 8 - 0.5, t, &in); if (d > worst) worst = d;
            if (i > a) len += sqrt(pow(q[0] - q[-3], 2) + pow(q[1] - q[-2], 2) + pow(q[2] - q[-1], 2)) / 8; }
        CHECK(worst < 0.75, "streamline %d (fibre %d): %.2f voxels off the axis", r, k, worst);
        CHECK(fabs(len - fibre_len(k)) < 6, "streamline %d (fibre %d): length %.1f, the fibre's %.1f", r, k, len, fibre_len(k));
        CHECK(F.fam[r] == (k >= 5 && k < 8 ? 'V' : 'H'), "streamline %d (fibre %d): family %c", r, k, F.fam[r]);
    }
    for (int k = 0; k < NF; k++) CHECK(seen[k] == 1, "fibre %d traced %d times", k, seen[k]);
    fib_free(&F);
    /* --presence-only: the axis from the presence's structure tensor; the same fibres, a looser tolerance (the ends, where the tube stops);
       the straight fibres come out as one line each, the radius-40 arc may break into pieces and run twice side by side (the 5^3 tensor on a 3-voxel tube) */
    char *ap[] = {"ufsm", "fiber-trace", root, key, out, "--box", "0,0,0,64,128,128", "--scaledown", "3", "--threads", "2", "--presence-only", "1"};
    CHECK(!cmd_fiber_trace(13, ap), "fiber-trace --presence-only");
    fib_set G = {0}; CHECK(!fib_load(out, &G), "load the presence-only streamlines"); int seen2[NF] = {0}; double w2 = 0, tl[NF] = {0};
    for (int r = 0; r < G.n; r++) {
        const int a = G.off[r], b = G.off[r + 1]; const float *m = G.pts + 3 * (size_t)((a + b) / 2); double t[3], best = 1e9; int k = -1, in;
        for (int j = 0; j < NF; j++) { const double d = fibre(j, (m[0] + 0.5) / 8 - 0.5, (m[1] + 0.5) / 8 - 0.5, (m[2] + 0.5) / 8 - 0.5, t, &in); if (d < best) { best = d; k = j; } }
        seen2[k]++; double len = 0;
        for (int i = a; i < b; i++) { const float *q = G.pts + 3 * (size_t)i; const double d = fibre(k, (q[0] + 0.5) / 8 - 0.5, (q[1] + 0.5) / 8 - 0.5, (q[2] + 0.5) / 8 - 0.5, t, &in); if (d > w2) w2 = d;
            if (i > a) len += sqrt(pow(q[0] - q[-3], 2) + pow(q[1] - q[-2], 2) + pow(q[2] - q[-1], 2)) / 8; }
        tl[k] += len;
    }
    CHECK(w2 < 1.5, "presence-only: %.2f voxels off the axis", w2);
    for (int k = 0; k < NF; k++) CHECK(tl[k] > 0.8 * fibre_len(k) && (k == 8 ? tl[k] < 2.2 * fibre_len(k) : tl[k] < fibre_len(k) + 6 && seen2[k] == 1), "presence-only: fibre %d traced %d times, length %.1f of %.1f", k, seen2[k], tl[k], fibre_len(k));
    fib_free(&G); for (int c = 0; c < 3; c++) free(v[c]);
    snprintf(p, sizeof p, "rm -rf %s", root); if (!fails && system(p)) fails++;
    printf(fails ? "test_fibertrace: %d FAILED\n" : "test_fibertrace: ok\n", fails);
    return fails ? 1 : 0;
}
