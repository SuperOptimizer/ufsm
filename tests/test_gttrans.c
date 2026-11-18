/* ufsm gt-trans (src/gttrans.c) on a synthetic stack: concentric sheets around the box centre, each a recto half (inner) and a verso
   half (outer), alternating packed (4 voxels apart, 38 um at 9.6) and spaced (12 voxels, 115 um) pairs. GT codes on the sheets (one
   turn per sheet), CT a gaussian ridge per sheet (merging where packed), rv = 200 on recto halves / 56 on verso halves, fibre H / V on
   the recto / verso halves. Expected, from OUT.tsv: every GT interval holds one V -> R and one V -> H flip (1 per turn in both gap
   buckets: the boundary between sheets counts even at contact), the CT peaks fall short where the sheets are packed. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_gt_trans(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
enum { D = 4, N = 240, S = 256 };

static int write_arr(const char *path, const uint8_t *a) {   /* one shard holds the whole array (lossless) */
    const int64_t sh[3] = {D, N, N}; z3w *w = z3w_create(path, sh, S, 0, 0, nullptr); if (!w) return 1;
    uint8_t *sb = calloc((size_t)S * S * S, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < N; y++) memcpy(sb + ((size_t)z * S + y) * S, a + ((size_t)z * N + y) * N, N);
    const int rc = z3w_write_shard(w, 0, 0, 0, sb, 4); free(sb); return rc | z3w_close(w);
}

int main(void) {
    char dir[] = "/tmp/ufsm-gttrans-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    const size_t n = (size_t)D * N * N; uint8_t *ct = calloc(n, 1), *cd = calloc(n, 1), *rv = calloc(n, 1), *fh = calloc(n, 1), *fv = calloc(n, 1);
    double rk[64]; int nk = 0; { double r = 14; for (int k = 0; nk < 64 && r < 112; k++) { rk[nk++] = r; r += k % 2 ? 12 : 4; } }   /* packed / spaced pairs */
    const double c = (N - 1) / 2.0;
    for (int z = 0; z < D; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const size_t i = ((size_t)z * N + y) * N + x; const double r = hypot(y - c, x - c); if (r > 116) continue;
        double v = 20; int best = -1; double bd = 1e9;
        for (int k = 0; k < nk; k++) { const double d = r - rk[k]; v += 200 * exp(-d * d / (2 * 0.8 * 0.8)); if (fabs(d) < bd) { bd = fabs(d); best = k; } }
        ct[i] = (uint8_t)fmin(250, v);
        const double d = r - rk[best];
        if (fabs(d) <= 1.0) cd[i] = (uint8_t)(1 + (18 * best) % 252);   /* GT: the sheet's code (one turn a sheet) */
        rv[i] = fabs(d) <= 2.0 ? (d < 0 ? 200 : 56) : 128;   /* recto inner half, verso outer half */
        fh[i] = fabs(d) <= 2.0 && d < 0 ? 200 : 10; fv[i] = fabs(d) <= 2.0 && d >= 0 ? 200 : 10;
    }
    char p[700];
    const char *nm[5] = {"ct", "cd", "rv", "fh", "fv"}; uint8_t *ar[5] = {ct, cd, rv, fh, fv};
    for (int k = 0; k < 5; k++) { snprintf(p, sizeof p, "%s/%s", dir, nm[k]); CHECK(!write_arr(p, ar[k]), "write %s", nm[k]); }
    char bx[700], out[700]; snprintf(bx, sizeof bx, "%s/boxes.tsv", dir); snprintf(out, sizeof out, "%s/out.tsv", dir);
    { FILE *f = fopen(bx, "w"); fprintf(f, "# name scroll z y x nz ny nx\nT1\tsyn\t0\t0\t0\t%d\t%d\t%d\t2\tsynthetic\nT2\tother\t0\t0\t0\t4\t4\t4\t2\tnot this scroll\n", D, N, N); fclose(f); }
    char *av[] = {"ufsm", "gt-trans", dir, "ct", "9.6", dir, "cd", "9.6", "ct", "9.6", bx, "syn", "9.6", out, "--nz", "2", "--nth", "90",
                  "--rv", dir, "rv", "--fh", dir, "fh", "--fv", dir, "fv", "--threads", "4"};
    CHECK(cmd_gt_trans((int)(sizeof av / sizeof *av), av) == 0, "gt-trans failed");
    /* the rows: per bucket (packed < 70 um, spaced >= 100 um) turns, CT peaks, V -> R and V -> H flips */
    double dq[2] = {0}, pk[2] = {0}, vr[2] = {0}, vh[2] = {0}; long cnt[2] = {0}; FILE *f = fopen(out, "r"); char line[512];
    if (f) { if (!fgets(line, sizeof line, f)) line[0] = 0;
        while (fgets(line, sizeof line, f)) { char b[32]; int z, npk, rvf, rvb, rvok, fbf, fbb, fbok, rvpk, fhpk; double rm, q, g;
            if (sscanf(line, "%31s %d %lf %lf %lf %d %d %d %d %d %d %d %d %d", b, &z, &rm, &q, &g, &npk, &rvf, &rvb, &rvok, &fbf, &fbb, &fbok, &rvpk, &fhpk) != 14) continue;
            const int k = g < 70 ? 0 : g >= 100 ? 1 : -1; if (k < 0 || !rvok || !fbok) continue;
            cnt[k]++; dq[k] += q; pk[k] += npk; vr[k] += rvf; vh[k] += fbf; }
        fclose(f); }
    for (int k = 0; k < 2; k++) printf("test_gttrans: %s: %ld intervals, CT %.3f, rv V>R %.3f, fib V>H %.3f per GT turn\n", k ? "spaced >= 100 um" : "packed < 70 um", cnt[k],
                                       dq[k] ? pk[k] / dq[k] : 0, dq[k] ? vr[k] / dq[k] : 0, dq[k] ? vh[k] / dq[k] : 0);
    CHECK(cnt[0] > 100 && cnt[1] > 100, "too few intervals: %ld packed, %ld spaced", cnt[0], cnt[1]);
    /* (+-0.1: a flip on an interval end, the sheet's own R | V interface at the GT centre, lands in either neighbour; the sum is 1 a turn) */
    for (int k = 0; k < 2; k++) { CHECK(fabs(vr[k] / dq[k] - 1) < 0.1, "rv V>R per turn %.3f (bucket %d)", vr[k] / dq[k], k); CHECK(fabs(vh[k] / dq[k] - 1) < 0.1, "fib V>H per turn %.3f (bucket %d)", vh[k] / dq[k], k); }
    CHECK(fabs((vr[0] + vr[1]) / (dq[0] + dq[1]) - 1) < 0.02, "rv V>R per turn overall %.3f", (vr[0] + vr[1]) / (dq[0] + dq[1]));
    /* the CT peaks merge on the packed sheets (ctgt's rule counts a peak on [ia, ib): a peak at the GT centre falls on either side, so
       only the packed deficit is checked) */
    CHECK(pk[0] / dq[0] < 0.6 && vr[0] / dq[0] > pk[0] / dq[0] + 0.3, "packed: CT %.3f vs rv %.3f per turn", pk[0] / dq[0], vr[0] / dq[0]);
    free(ct); free(cd); free(rv); free(fh); free(fv);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) fprintf(stderr, "(could not remove %s)\n", dir);
    if (fails) { fprintf(stderr, "test_gttrans: %d failure(s)\n", fails); return 1; }
    printf("test_gttrans: OK\n");
    return 0;
}
