/* bench_votes <root> <side-key> [z,y,x ...] [--n 256] : how the lossless volcomp codec stores recto / verso VOTE volumes (src/votes.h).
   Each cube (n^3 at z,y,x of a phase-raster --pair 2 side array) is set A; sets B and C are A seen through a smooth displacement
   (+-2 voxels, a few hundred voxels' period: the same wraps traced again by another author, a voxel or two off) over a smooth
   ~60% / ~40% of the cube (each author traced part of the scroll). Votes = A + B + C. Encoded per 128^3 chunk (q 0):
     single     A alone (today's side array: 0 / 1 verso / 2 recto)
     diag       the votes in src/votes.h's diagonal code
     nibble     the votes as r << 4 | v
     decoded    the unanimous decode of the votes (0 / 1 / 2 again: what the sampler reads)
   Prints bytes, bits per labelled voxel, encode / decode MB/s, and the vote classes' shares. */
#include "zarr3.h"
#include "store.h"
#include "votes.h"
#include "volcomp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { double az[3], ph[3], lam[3]; } field;   /* d(p) = sum_k a_k sin(2 pi p_k / lam_k + ph_k) style, per axis */
static void field_init(field *f, unsigned seed, double amp) {
    srand(seed);
    for (int d = 0; d < 3; d++) { f->az[d] = amp * (0.5 + 0.5 * rand() / (double)RAND_MAX); f->ph[d] = 6.283 * rand() / (double)RAND_MAX; f->lam[d] = 160 + 240.0 * rand() / RAND_MAX; }
}
static double field_at(const field *f, int d, double z, double y, double x) {   /* displacement along axis d */
    const double p[3] = {z, y, x}; const int a = (d + 1) % 3, b = (d + 2) % 3;
    return f->az[d] * sin(6.283185 * p[a] / f->lam[a] + f->ph[d]) * cos(6.283185 * p[b] / f->lam[b] + f->ph[(d + 1) % 3]);
}

static void encode_all(const uint8_t *vol, int n, size_t *bytes, double *tenc, double *tdec) {
    uint8_t *chunk = malloc(VOLCOMP_CHUNK_VOXELS), *enc = malloc(VOLCOMP_ENCODE_BOUND), *dec = malloc(VOLCOMP_CHUNK_VOXELS);
    const int nc = n / 128;
    for (int cz = 0; cz < nc; cz++) for (int cy = 0; cy < nc; cy++) for (int cx = 0; cx < nc; cx++) {
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++)
            memcpy(chunk + ((size_t)z * 128 + y) * 128, vol + (((size_t)(cz * 128 + z) * n + cy * 128 + y) * n + cx * 128), 128);
        size_t m = 0; double t0 = now();
        if (volcomp_encode(chunk, 0.f, enc, VOLCOMP_ENCODE_BOUND, &m) != VOLCOMP_OK) { fprintf(stderr, "encode failed\n"); exit(1); }
        double t1 = now();
        if (volcomp_decode(enc, m, dec, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK || memcmp(dec, chunk, VOLCOMP_CHUNK_VOXELS)) { fprintf(stderr, "decode mismatch\n"); exit(1); }
        double t2 = now();
        *bytes += m; *tenc += t1 - t0; *tdec += t2 - t1;
    }
    free(chunk); free(enc); free(dec);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: bench_votes <root> <side-key> [z,y,x ...] [--n 256]\n"); return 2; }
    int n = 256; long long cubes[16][3]; int ncubes = 0;
    for (int a = 3; a < argc; a++) {
        if (!strcmp(argv[a], "--n") && a + 1 < argc) { n = atoi(argv[++a]); continue; }
        if (ncubes < 16 && sscanf(argv[a], "%lld,%lld,%lld", &cubes[ncubes][0], &cubes[ncubes][1], &cubes[ncubes][2]) == 3) ncubes++;
    }
    if (n % 128) { fprintf(stderr, "--n: a multiple of 128\n"); return 2; }
    if (!ncubes) { const long long d[4][3] = {{24064, 8448, 7680}, {24576, 9216, 8448}, {25088, 7936, 6912}, {23808, 9984, 9216}}; memcpy(cubes, d, sizeof d); ncubes = 4; }
    z3 *z = z3_open(store_open(argv[1]), argv[2], nullptr);
    if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const size_t N = (size_t)n * n * n;
    uint8_t *A = malloc(N), *V[3] = {malloc(N), malloc(N), malloc(N)}, *vol[4] = {malloc(N), malloc(N), malloc(N), malloc(N)};
    const char *names[4] = {"single", "diag", "nibble", "decoded"};
    size_t bytes[4] = {0}, lab = 0, cls[256] = {0}; double te[4] = {0}, td[4] = {0};
    uint8_t lut[256]; vote_side_lut(VOTE_UNANIMOUS, 1, lut);
    for (int c = 0; c < ncubes; c++) {
        const int64_t o[3] = {cubes[c][0], cubes[c][1], cubes[c][2]}, nn[3] = {n, n, n};
        if (z3_read(z, o, nn, A, 8)) { fprintf(stderr, "read %lld,%lld,%lld: %s\n", cubes[c][0], cubes[c][1], cubes[c][2], z3_error()); return 1; }
        memcpy(V[0], A, N);
        for (int s = 1; s < 3; s++) {   /* sets B, C: displaced copies over part of the cube */
            field f, cov; field_init(&f, 11u * s + 7u * c, 2.0); field_init(&cov, 101u * s + 13u * c, 1.0);
            const double keep = s == 1 ? 0.25 : -0.25;   /* coverage threshold: ~60% / ~40% */
            for (int zz = 0; zz < n; zz++) for (int yy = 0; yy < n; yy++) for (int xx = 0; xx < n; xx++) {
                const size_t k = ((size_t)zz * n + yy) * n + xx;
                if (field_at(&cov, 0, zz, yy, xx) + field_at(&cov, 1, zz, yy, xx) < -keep) { V[s][k] = 0; continue; }
                int q[3] = {zz + (int)lround(field_at(&f, 0, zz, yy, xx)), yy + (int)lround(field_at(&f, 1, zz, yy, xx)), xx + (int)lround(field_at(&f, 2, zz, yy, xx))};
                for (int d = 0; d < 3; d++) q[d] = q[d] < 0 ? 0 : q[d] >= n ? n - 1 : q[d];
                V[s][k] = A[((size_t)q[0] * n + q[1]) * n + q[2]];
            }
        }
        for (size_t k = 0; k < N; k++) {
            int r = 0, v = 0;
            for (int s = 0; s < 3; s++) { r += V[s][k] == 2; v += V[s][k] == 1; }
            const uint8_t dc = vote_code(r, v);
            vol[0][k] = A[k]; vol[1][k] = dc; vol[2][k] = (uint8_t)(r << 4 | v); vol[3][k] = lut[dc];
            cls[dc]++; lab += r + v > 0;
        }
        for (int e = 0; e < 4; e++) encode_all(vol[e], n, &bytes[e], &te[e], &td[e]);
    }
    const double mb = (double)N * ncubes / 1e6;
    printf("%d cubes of %d^3 (%.0f M voxels), %.1f%% with votes\n", ncubes, n, mb, 100.0 * lab / ((double)N * ncubes));
    for (int e = 0; e < 4; e++)
        printf("  %-8s %10zu bytes  %.3f bits/voxel  %.3f bits/labelled voxel  encode %.0f MB/s  decode %.0f MB/s\n", names[e], bytes[e],
               8.0 * bytes[e] / ((double)N * ncubes), 8.0 * bytes[e] / (double)(lab ? lab : 1), mb / te[e], mb / td[e]);
    printf("  vote classes (share of voxels with votes):");
    for (int c = 1; c < 256; c++) if (cls[c] * 1000 >= lab) { int r, v; vote_split((uint8_t)c, &r, &v); printf(" (%d,%d) %.1f%%", r, v, 100.0 * cls[c] / lab); }
    size_t conf = 0; for (int c = 1; c < 256; c++) { int r, v; vote_split((uint8_t)c, &r, &v); if (r && v) conf += cls[c]; }
    printf("\n  conflicting (r > 0 and v > 0): %.2f%% of voxels with votes\n", 100.0 * conf / (lab ? lab : 1));
    z3_close(z);
    return 0;
}
