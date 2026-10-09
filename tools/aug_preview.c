/* Before/after slices of every scanner-domain op (src/scan_augment.h) and the affine warp on one real CT cube, at mild /
   default / aggressive strength. Each PNG: original axial (z = P/2) | augmented axial | augmented y = P/2 (z down) |
   supervision (white = supervised, grey = newly ignored, black = CT 0). Prints the single-thread cost of each op. */
#include "zarr3.h"
#include "scan_augment.h"
#include "spatial_augment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

static void be32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static int chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4);
    uLong crc = crc32(crc32(0, (const Bytef *)type, 4), d, n); uint8_t c[4]; be32(c, (uint32_t)crc);
    return fwrite(h, 1, 8, f) != 8 || (n && fwrite(d, 1, n, f) != n) || fwrite(c, 1, 4, f) != 4;
}
static int write_png(const char *path, const uint8_t *img, int w, int h) {   /* 8-bit grey, filter 0 */
    size_t raw = (size_t)(w + 1) * h; uint8_t *r = malloc(raw); uLongf zn = compressBound(raw); uint8_t *z = malloc(zn);
    if (!r || !z) return 1;
    for (int y = 0; y < h; y++) { r[(size_t)y * (w + 1)] = 0; memcpy(r + (size_t)y * (w + 1) + 1, img + (size_t)y * w, w); }
    int rc = compress2(z, &zn, r, raw, 6) != Z_OK;
    FILE *f = fopen(path, "wb");
    uint8_t ihdr[13]; be32(ihdr, w); be32(ihdr + 4, h); ihdr[8] = 8; ihdr[9] = 0; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    rc |= !f || fwrite("\x89PNG\r\n\x1a\n", 1, 8, f) != 8 || chunk(f, "IHDR", ihdr, 13) || chunk(f, "IDAT", z, (uint32_t)zn) || chunk(f, "IEND", nullptr, 0);
    if (f) rc |= fclose(f) != 0;
    free(r); free(z); return rc;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int panels(const char *path, const uint8_t *ct0, const uint8_t *ct, const uint8_t *ign0, const uint8_t *ign, int P, int S) {
    int w = 4 * S; uint8_t *img = calloc((size_t)w * S, 1); if (!img) return 1;
    const size_t P2 = (size_t)P * P; const int z = P / 2, yc = P / 2;
    for (int a = 0; a < S; a++) for (int b = 0; b < S; b++) {
        int u = a * P / S, v = b * P / S; size_t ax = (size_t)z * P2 + (size_t)u * P + v, cor = (size_t)u * P2 + (size_t)yc * P + v;
        uint8_t *row = img + (size_t)a * w;
        row[b] = ct0[ax]; row[S + b] = ct[ax]; row[2 * S + b] = ct[cor];
        row[3 * S + b] = !ct[ax] ? 0 : ign[ax] && !ign0[ax] ? 110 : ign[ax] ? 50 : 255;
    }
    int rc = write_png(path, img, w, S); free(img); return rc;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: aug_preview <zarr-group> <out-dir> [z y x (level 0, default 48128 17408 16384)] [P=512] [level=0]\n"); return 2; }
    int64_t o[3] = {48128, 17408, 16384};
    if (argc >= 6) for (int d = 0; d < 3; d++) o[d] = atoll(argv[3 + d]);
    int P = argc >= 7 ? atoi(argv[6]) : 512, lev = argc >= 8 ? atoi(argv[7]) : 0, S = P > 512 ? 512 : P;
    if (P < 16 || P > 1024) return 2;
    store *st = store_open(argv[1]); char key[16]; snprintf(key, sizeof key, "%d", lev);
    z3 *z = st ? z3_open(st, key, nullptr) : nullptr;
    if (!z) { fprintf(stderr, "open: %s\n", z3_error()); return 1; }
    const size_t n = (size_t)P * P * P;
    int64_t oo[3] = {o[0] >> lev, o[1] >> lev, o[2] >> lev}, nn[3] = {P, P, P};
    uint8_t *ct0 = malloc(n), *ct = malloc(n), *ign0 = calloc(n, 1), *ign = malloc(n), *tgt = calloc(n, 1), *scratch = malloc(5 * n);
    float *cy = malloc(P * sizeof *cy), *cx = malloc(P * sizeof *cx);
    if (!ct0 || !ct || !ign0 || !ign || !tgt || !scratch || !cy || !cx) return 1;
    if (z3_read(z, oo, nn, ct0, 0)) { fprintf(stderr, "read: %s\n", z3_error()); return 1; }
    const int *chunkdim = z3_meta_of(z)->chunk;
    const float strengths[3] = {.5f, 1, 2}; const char *sname[3] = {"mild", "default", "aggressive"};
    char path[1400];
    snprintf(path, sizeof path, "%s/original.png", argv[2]);
    if (panels(path, ct0, ct0, ign0, ign0, P, S)) { fprintf(stderr, "write %s failed\n", path); return 1; }
    printf("%-10s %10s %10s %10s   (single-thread ms on %d^3, incl. the float conversion)\n", "op", sname[0], sname[1], sname[2], P);
    for (int op = 0; op <= SA_N + 1; op++) {   /* SA_N: every op at p .3; SA_N + 1: affine */
        const char *name = op < SA_N ? scan_aug_names[op] : op == SA_N ? "combo" : "affine";
        double ms[3];
        for (int si = 0; si < 3; si++) {
            memcpy(ct, ct0, n); memset(ign, 0, n);
            for (int d = 0; d < P; d++) { cy[d] = -1e4f; cx[d] = -1e4f; }
            double t0 = now();
            if (op <= SA_N) {
                scan_aug_cfg c = {.strength = strengths[si], .drop_ignore = 1};
                if (op < SA_N) c.p[op] = 1; else for (int i = 0; i < SA_N; i++) c.p[i] = .3f;
                scan_aug_apply(&c, 11 + (uint64_t)si, ct, tgt, 1, ign, P, oo, cy, cx, 0, chunkdim, 1, scratch);
            } else {   /* anisotropic scale + shear at the trainer's default magnitudes x strength, trilinear pull */
                spatial_aug a; spatial_aug_make(&a, 5, P, 0, 0, 0, 0); spatial_aug_affine(&a, 5, .1 * strengths[si], .06 * strengths[si]);
                float *table = calloc(6 * (size_t)P, sizeof *table); spatial_aug_tables(&a, table);
                for (int zz = 0; zz < P; zz++) for (int yy = 0; yy < P; yy++) for (int xx = 0; xx < P; xx++) {
                    float u[3], jac[3]; spatial_aug_pull(&a, table, zz, yy, xx, u, jac);
                    size_t k = ((size_t)zz * P + yy) * P + xx; int inside = 1, b[3]; float f[3];
                    for (int d = 0; d < 3; d++) { inside &= u[d] >= 0 && u[d] <= P - 1; b[d] = (int)u[d]; if (b[d] >= P - 1) b[d] = P - 2; if (b[d] < 0) b[d] = 0; f[d] = u[d] - b[d]; }
                    if (!inside) { ct[k] = 0; ign[k] = 1; continue; }
                    float v = 0;
                    for (int c = 0; c < 8; c++) { int dz = c >> 2, dy = (c >> 1) & 1, dx = c & 1; v += (dz ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dx ? f[2] : 1 - f[2]) * ct0[((size_t)(b[0] + dz) * P + b[1] + dy) * P + b[2] + dx]; }
                    ct[k] = (uint8_t)lrintf(v);
                }
                free(table);
            }
            ms[si] = (now() - t0) * 1e3;
            snprintf(path, sizeof path, "%s/%02d-%s-%s.png", argv[2], op, name, sname[si]);
            if (panels(path, ct0, ct, ign0, ign, P, S)) { fprintf(stderr, "write %s failed\n", path); return 1; }
        }
        printf("%-10s %10.0f %10.0f %10.0f\n", name, ms[0], ms[1], ms[2]);
    }
    z3_close(z); store_close(st);
    return 0;
}
