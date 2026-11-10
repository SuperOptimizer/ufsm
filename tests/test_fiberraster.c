/* ufsm fiber-raster on three UFIB fibres in a 128 x 256 x 256 level-1 grid (points are level-0 voxels: (p + 0.5) 2 - 0.5): an H fibre
   along x, a V fibre along z that passes 30 voxels from it, and an H fibre that crosses a second V fibre. Read back: every voxel
   within the radius of the lone H fibre is 254 in plane H and 0 in plane V (and the reverse on the V fibre), the guard ring and
   everything away from the fibres are 255, the crossing is 255 in both planes, and a second run with --zband writes only that band.
   With --ct (a CT of 100 with an air block of 10): air away from fibres is 0 in both planes, papyrus stays 255. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"

int cmd_fiber_raster(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
enum { NZ = 128, NY = 256, NX = 256, NP = 120 };
static uint8_t *rd(const char *root, const char *key) {
    z3 *z = z3_open(store_open(root), key, nullptr); if (!z) return nullptr; const int64_t o[3] = {0, 0, 0}, n[3] = {NZ, NY, NX}; uint8_t *b = malloc((size_t)NZ * NY * NX);
    if (z3_read(z, o, n, b, 2)) { free(b); b = nullptr; } z3_close(z); return b;
}
/* level-1 axis of fibre k at parameter t in [0, 1] */
static void axis(int k, double t, double p[3]) {
    if (k == 0) { p[0] = 40; p[1] = 60; p[2] = 20 + 200 * t; }          /* H along x */
    else if (k == 1) { p[0] = 10 + 100 * t; p[1] = 90; p[2] = 120; }    /* V along z, 30 voxels from fibre 0 */
    else if (k == 2) { p[0] = 80; p[1] = 200; p[2] = 20 + 200 * t; }    /* H along x .. */
    else { p[0] = 30 + 90 * t; p[1] = 200; p[2] = 150; }                /* .. crossed by a V fibre at (80, 200, 150) */
}
static double dist(int k, double z, double y, double x) { double best = 1e9, p[3]; for (int i = 0; i <= 800; i++) { axis(k, i / 800.0, p); const double d = sqrt(pow(z - p[0], 2) + pow(y - p[1], 2) + pow(x - p[2], 2)); if (d < best) best = d; } return best; }

int main(void) {
    char root[] = "/tmp/ufsm-fiberraster-XXXXXX"; if (!mkdtemp(root)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char fb[400], out[400], out2[400], ctd[400], p[900]; snprintf(fb, sizeof fb, "%s/f.fib", root); snprintf(out, sizeof out, "%s/lab", root); snprintf(out2, sizeof out2, "%s/lab2", root); snprintf(ctd, sizeof ctd, "%s/ct", root);
    { fib_set F = {0}; F.n = 4; F.npts = 4 * NP; int off[5]; char fam[5] = {'H', 'V', 'H', 'V', 0}; float *pt = malloc(sizeof(float) * 3 * 4 * NP); F.off = off; F.fam = fam; F.pts = pt;
      for (int r = 0; r <= 4; r++) off[r] = r * NP;
      for (int r = 0; r < 4; r++) for (int i = 0; i < NP; i++) { double a[3]; axis(r, i / (double)(NP - 1), a); for (int d = 0; d < 3; d++) pt[3 * (r * NP + i) + d] = (float)((a[d] + 0.5) * 2 - 0.5); }
      CHECK(!fib_save(fb, &F), "write the fibres"); free(pt); }
    char *av[] = {"ufsm", "fiber-raster", fb, out, "--um", "9.6", "--shape", "128,256,256", "--level", "1", "--radius", "2", "--guard", "3", "--shard", "128", "--threads", "3"};
    CHECK(!cmd_fiber_raster(18, av), "fiber-raster");
    uint8_t *H = rd(out, "H/9.6"), *V = rd(out, "V/9.6"); CHECK(H && V, "read the planes back"); if (!H || !V) return 1;
    long bad = 0, core = 0, ring = 0, far = 0, cross = 0;
    for (int z = 0; z < NZ; z += 1) for (int y = 40; y < 230; y += 1) for (int x = 0; x < NX; x += 3) {
        const size_t i = ((size_t)z * NY + y) * NX + x; double d[4]; int near = 0; for (int k = 0; k < 4; k++) { d[k] = abs(z - (k == 0 ? 40 : k == 2 ? 80 : z)) > 8 || abs(y - (k < 2 ? (k ? 90 : 60) : 200)) > 8 ? 99 : dist(k, z, y, x); near += d[k] < 5.6; }
        const double dh = d[0] < d[2] ? d[0] : d[2], dv = d[1] < d[3] ? d[1] : d[3];
        if (dh < 1.8 && dv > 2.2) { core++; bad += H[i] != 254 || V[i] != 0; }
        else if (dv < 1.8 && dh > 2.2) { core++; bad += V[i] != 254 || H[i] != 0; }
        else if (dh < 1.8 && dv < 1.8) { cross++; bad += H[i] != 255 || V[i] != 255; }
        else if (dh > 2.2 && dv > 2.2) { if (near) ring++; else far++; bad += H[i] != 255 || V[i] != 255; }
    }
    CHECK(!bad && core > 800 && ring > 3000 && far > 100000 && cross > 3, "labels: %ld wrong of %ld core, %ld ring, %ld far, %ld crossing voxels", bad, core, ring, far, cross);
    free(H); free(V);
    /* --ct: air -> 0 away from fibres; --zband: only shard row 0 is written */
    { const int64_t shp[3] = {NZ, NY, NX}; snprintf(p, sizeof p, "%s/9.6", ctd); const double um = 9.6; z3w *w = z3w_create(p, shp, 128, 0, 0, nullptr); CHECK(w != nullptr, "CT"); uint8_t *b = malloc((size_t)128 * 128 * 128);
      for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) { memset(b, sy == 0 && sx == 0 ? 10 : 100, (size_t)128 * 128 * 128); CHECK(!z3w_write_shard(w, 0, sy, sx, b, 1), "CT shard"); }
      free(b); CHECK(!z3w_close(w) && !z3w_write_group(ctd, &um, 1, "ct", nullptr), "CT close"); }
    char *aw[] = {"ufsm", "fiber-raster", fb, out2, "--um", "9.6", "--ct", ctd, "9.6", "--iso", "64", "--level", "1", "--shard", "128", "--zband", "0,100", "--threads", "2"};
    CHECK(!cmd_fiber_raster(19, aw), "fiber-raster --ct --zband");
    H = rd(out2, "H/9.6"); V = rd(out2, "V/9.6"); CHECK(H && V, "read the --ct planes back"); if (!H || !V) return 1;
    { const size_t air = ((size_t)100 * NY + 20) * NX + 20, pap = ((size_t)100 * NY + 20) * NX + 200, onh = ((size_t)40 * NY + 60) * NX + 60, onv = ((size_t)60 * NY + 90) * NX + 120;
      CHECK(H[air] == 0 && V[air] == 0, "air away from fibres: H %d V %d, want 0 0", H[air], V[air]);
      CHECK(H[pap] == 255 && V[pap] == 255, "papyrus away from fibres: H %d V %d, want 255 255", H[pap], V[pap]);
      CHECK(H[onh] == 254 && V[onh] == 0 && V[onv] == 254 && H[onv] == 0, "fibres over air keep their labels (H %d %d, V %d %d)", H[onh], V[onh], V[onv], H[onv]); }
    free(H); free(V);
    snprintf(p, sizeof p, "rm -rf %s", root); if (!fails && system(p)) fails++;
    printf(fails ? "test_fiberraster: %d FAILED\n" : "test_fiberraster: ok\n", fails);
    return fails ? 1 : 0;
}
