/* ufsm recto-sp: the GPU path (src/nn_rectosp.cu, --gpu 0) against the CPU path (--gpu -1), byte for byte (sp and rsp), on a
   synthetic spiral with a noisy recto "probability" (irregular peaks, plateaus, NaN-free but clipped): 200 field rows (4 shard-row
   blocks of 64 rows and a partial one, so the source ring wraps and refills), box origin off the 512 / 128 grids, three flag sets:
   the unroll's (--half-um 400 --half2-um 800 --blend 0.3, stride 4), stride 3 (row blocks whose stride grid restarts off the block
   start) with --rows and the source decoded on the host (UFSM_RECTOSP_DEVDEC=0), and a wide --sigma / --sep-um. The source is
   lossy (q 8, as the m7 stores), so the other sets decode it on the GPU decoder into the ring. */
#include "fit.h"
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_recto_sp(int argc, char **argv);
extern int g_rectosp_gpu_blocks;
extern long long g_rectosp_dev_chunks, g_rectosp_ring_chunks;
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 200, HH = 112, WW = 120, H2 = 2, SHF = 128, SHS = 256 };
static const double CY = 118, CX = 126, SP = 15.3;

static int write_arr(const char *path, const uint8_t *a, const int64_t sh[3], int S, float q) {   /* a whole array, shard by shard (q > 0: lossy) */
    z3w *w = z3w_create(path, sh, S, q, 0, nullptr); if (!w) return 1;
    uint8_t *sb = malloc((size_t)S * S * S); int rc = 0;
    for (int64_t z = 0; z * S < sh[0]; z++) for (int64_t y = 0; y * S < sh[1]; y++) for (int64_t x = 0; x * S < sh[2]; x++) {
        memset(sb, 0, (size_t)S * S * S);
        for (int64_t a2 = 0; a2 < S && z * S + a2 < sh[0]; a2++) for (int64_t b = 0; b < S && y * S + b < sh[1]; b++) {
            const int64_t x0 = x * S, nx = x0 + S <= sh[2] ? S : sh[2] - x0;
            memcpy(sb + ((size_t)a2 * S + b) * S, a + ((size_t)(z * S + a2) * sh[1] + y * S + b) * sh[2] + x0, nx);
        }
        rc |= z3w_write_shard(w, z, y, x, sb, 4);
    }
    free(sb); return rc | z3w_close(w);
}
static int read_all(const char *dir, const char *key, uint8_t *b, size_t n, const int64_t sh[3]) {
    z3 *z = z3_open(store_open(dir), key, nullptr); if (!z) return 1;
    const int64_t o[3] = {0, 0, 0}; const int rc = z3_read(z, o, sh, b, 4); z3_close(z); (void)n; return rc;
}

int main(void) {
    setenv("UFSM_GPU_DECODE", "1", 1); setenv("UFSM_GPU_DECODE_DEV", "0", 1);   /* (the CPU path's reads decode on it too: the same bytes) */
    char dir[] = "/tmp/ufsm-rectogpu-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char p[700], fd[600]; snprintf(fd, sizeof fd, "%s/fields", dir);
    /* fields: box origin (native) 70, 6, 10 (off every grid); material r 30..110 (native); sp 15.3 native; radial normals */
    const long long org[3] = {70, 6, 10};
    const size_t nc = (size_t)D * HH * WW; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(nc, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < HH; y++) for (int x = 0; x < WW; x++) {
        const size_t i = ((size_t)z * HH + y) * WW + x; const double dy = org[1] + (y + 0.5) * H2 - CY, dx = org[2] + (x + 0.5) * H2 - CX, r = hypot(dy, dx);
        const double tw = 0.15 * sin(z * 0.05), ny = r > 0 ? dy / r : 0, nx = r > 0 ? dx / r : 1, nz = tw, l = sqrt(1 + nz * nz);   /* (a little tilt in z) */
        const double n3[3] = {nz / l, ny / l, nx / l};
        a[0][i] = a[1][i] = 128; for (int t = 0; t < 3; t++) a[2 + t][i] = (uint8_t)lround(255 * n3[t] * n3[t]);
        a[5][i] = (uint8_t)lround(255 * (0.5 + n3[0] * n3[1])); a[6][i] = (uint8_t)lround(255 * (0.5 + n3[0] * n3[2])); a[7][i] = (uint8_t)lround(255 * (0.5 + n3[1] * n3[2]));
        a[8][i] = (uint8_t)(SP * 8 / H2 + (x + y + z) % 7 - 3); a[9][i] = r >= 30 && r <= 110 && (x * 7 + y * 3 + z) % 23 ? 255 : 0;
    }
    const int64_t fsh[3] = {D, HH, WW}; const char *pn[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    for (int k = 0; k < 10; k++) { snprintf(p, sizeof p, "%s/%s", fd, pn[k]); CHECK(!write_arr(p, a[k], fsh, SHF, 0), "write plane %s", pn[k]); }
    {   snprintf(p, sizeof p, "%s/fields.json", fd); FILE *f = fopen(p, "w");
        fprintf(f, "{\"format\":\"zarr\",\"origin\":[%lld,%lld,%lld],\"shape\":[%d,%d,%d],\"h\":%d,\"um\":9.6,\"axis\":[", org[0], org[1], org[2], D, HH, WW, H2);
        for (int i = 0; i <= D * H2; i++) fprintf(f, "%s[%g,%g]", i ? "," : "", CY, CX);
        fprintf(f, "]}\n"); fclose(f); }
    /* the source: native grid covering the box and more (z 0 .. 480), a recto peak per sheet, noise, some plateaus (clipped at 250),
       zeros between the peaks on some rows */
    const int64_t sh[3] = {480, 250, 260}; uint8_t *v = malloc((size_t)sh[0] * sh[1] * sh[2]); srand(11);
    for (int64_t z = 0; z < sh[0]; z++) for (int64_t y = 0; y < sh[1]; y++) for (int64_t x = 0; x < sh[2]; x++) {
        const double ry = y + 0.5 - CY, rx = x + 0.5 - CX, w = hypot(ry, rx) / SP - atan2(ry, rx) / (2 * M_PI) + 0.02 * sin(z * 0.11), f = w - floor(w + 0.5), d = fabs(f) * SP;
        const double val = 8 + 260 * exp(-d * d / 4.0) * (0.7 + 0.3 * sin(z * 0.07 + x * 0.05)) + rand() % 30;
        const int zero = (z % 200 < 120 && d > 3) || (z >= 300 && z < 330);   /* (zero runs and an empty slab: the kernel's exact-zero shortcuts) */
        v[((size_t)z * sh[1] + y) * sh[2] + x] = zero ? 0 : (uint8_t)(val > 250 ? 250 : val);
    }
    snprintf(p, sizeof p, "%s/rv.zarr", dir); CHECK(!write_arr(p, v, sh, SHS, 8), "write recto");   /* (lossy q 8 as the m7 stores: the GPU decoder takes its chunks) */
    free(v);
    const char *sets[3][8] = {{"--stride", "4", "--half-um", "400", "--half2-um", "800", "--blend", "0.3"},
                              {"--stride", "3", "--half-um", "300", "--rows", "10,190", "--min-int", "2"},
                              {"--stride", "5", "--half-um", "250", "--sigma", "1.7", "--sep-um", "35"}};
    uint8_t *b1 = malloc(nc), *b2 = malloc(nc);
    for (int s = 0; s < 3; s++) {
        char o1[700], o2[700]; snprintf(o1, sizeof o1, "%s/cpu%d", dir, s); snprintf(o2, sizeof o2, "%s/gpu%d", dir, s);
        for (int g = 0; g < 2; g++) {
            char *av[20] = {"ufsm", "recto-sp", fd, g ? o2 : o1, "--src", dir, "rv.zarr", "9.6", "--threads", "4", "--gpu", g ? "0" : "-1"}; int ac = 12;
            for (int k = 0; k < 8; k++) av[ac++] = (char *)sets[s][k];
            if (g && s == 1) setenv("UFSM_RECTOSP_DEVDEC", "0", 1);   /* (set 1: the source decoded on the host and uploaded) */
            CHECK(cmd_recto_sp(ac, av) == 0, "recto-sp set %d %s", s, g ? "gpu" : "cpu");
            unsetenv("UFSM_RECTOSP_DEVDEC");
            CHECK(g ? g_rectosp_gpu_blocks >= 3 : g_rectosp_gpu_blocks == 0, "set %d %s: %d blocks on the GPU", s, g ? "gpu" : "cpu", g_rectosp_gpu_blocks);
            if (g) { CHECK(s == 1 ? g_rectosp_ring_chunks == -1 : g_rectosp_ring_chunks > 0, "set %d: %lld source chunks into the ring on the device", s, g_rectosp_ring_chunks);
                     printf("test_recto_gpu: set %d: %lld source chunks via the device ring, %lld of them on the GPU decoder (it may hand chunks to the CPU)\n", s, g_rectosp_ring_chunks, g_rectosp_dev_chunks); }
        }
        for (int k = 0; k < 2; k++) {
            const char *key = k ? "rsp" : "sp";
            CHECK(!read_all(o1, key, b1, nc, fsh) && !read_all(o2, key, b2, nc, fsh), "read %s set %d", key, s);
            size_t nd = 0, nz = 0, first = (size_t)-1; for (size_t i = 0; i < nc; i++) { nz += b1[i] != 0; if (b1[i] != b2[i]) { if (!nd) first = i; nd++; } }
            printf("test_recto_gpu: set %d %s: %zu of %zu cells differ (%zu nonzero on the CPU)\n", s, key, nd, nc, nz);
            CHECK(nd == 0, "set %d %s: %zu cells differ, first %zu (cpu %d gpu %d)", s, key, nd, first, nd ? b1[first] : 0, nd ? b2[first] : 0);
            if (k) CHECK(nz > nc / 40, "set %d: only %zu recto spacings", s, nz);
        }
    }
    free(b1); free(b2); for (int k = 0; k < 10; k++) free(a[k]);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) fprintf(stderr, "(could not remove %s)\n", dir);
    if (fails) { fprintf(stderr, "test_recto_gpu: %d failure(s)\n", fails); return 1; }
    printf("test_recto_gpu: OK\n");
    return 0;
}
