/* test_gt_gate (2026-11-14, ops audit #17): ufsm gt-gate, the training-label gate of the GT whose sheets do not follow the CT, had no
   test. Fixture (128^3, axis at y = x = 64): GT sheets on three cylinders (r 20, 32, 44; codes 1..252 along theta), a prob volume with
   radial ridges on the same cylinders where x < 64 and nothing where x >= 64. With --cell 64: the four cells with x >= 64 have no
   support and are gated (255), the other four are kept (0); the printed share of gated sheet voxels is about half. CPU. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

int cmd_gt_gate(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
enum { N = 128 };
static const double R[3] = {20, 32, 44};

static int write_arr(const char *path, const uint8_t *v) {
    const int64_t sh[3] = {N, N, N}; z3w *w = z3w_create(path, sh, 128, 0, 0, nullptr);
    return !w || z3w_write_shard(w, 0, 0, 0, v, 1) || z3w_close(w);
}

int main(void) {
    const char *t = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp"; char D[1024]; snprintf(D, sizeof D, "%s/ttg.%d", t, (int)getpid()); mkdir(D, 0755);
    uint8_t *gt = calloc((size_t)N * N * N, 1), *pb = calloc((size_t)N * N * N, 1); long nsheet = 0, nright = 0;
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const double dy = y + 0.5 - 64, dx = x + 0.5 - 64, r = hypot(dy, dx), th = atan2(dy, dx); const size_t i = ((size_t)z * N + y) * N + x;
        double best = 99; for (int k = 0; k < 3; k++) best = fmin(best, fabs(r - R[k]));
        if (best < 0.5) { gt[i] = (uint8_t)(1 + (int)floor((th + M_PI) / (2 * M_PI) * 251.999)); nsheet++; nright += x >= 64; }
        if (x < 64) pb[i] = (uint8_t)lrint(250 * exp(-best * best / 4.0));   /* a ridge of width ~2 voxels on each cylinder */
    }
    char p[1200], q[1200];
    snprintf(p, sizeof p, "%s/gt/1", D); CHECK(!write_arr(p, gt), "write the GT");
    snprintf(p, sizeof p, "%s/prob/1", D); CHECK(!write_arr(p, pb), "write the prob");
    snprintf(p, sizeof p, "%s/axis.json", D); FILE *f = fopen(p, "w");
    fprintf(f, "{\"control_points\": [{\"x\": 64, \"y\": 64, \"z\": 0}, {\"x\": 64, \"y\": 64, \"z\": 127}]}\n"); fclose(f);
    snprintf(q, sizeof q, "%s/out", D);
    char res[1200]; snprintf(res, sizeof res, "%s/stdout.txt", D);
    fflush(stdout);
    const pid_t c = fork();
    if (c == 0) {
        const int fd = open(res, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) _exit(9); dup2(fd, 1);
        char *av[] = {"ufsm", "gt-gate", "--gt", D, "--gt-key", "gt/1", "--prob", D, "--prob-key", "prob/1", "--axis", p, "--out", q, "--um", "1",
                      "--cell", "64", "--min-n", "200", "--jobs", "2", "--dir", "radial", nullptr};
        const int r = cmd_gt_gate(22, av); fflush(stdout); _exit(r);   /* _exit skips the stdio flush */
    }
    int st = 0; waitpid(c, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "gt-gate ran (status %d)", st);
    f = fopen(res, "r"); char line[1024]; long long sh = -1, cells = -1, gc = -1; double share = -1;
    while (f && fgets(line, sizeof line, f)) { fputs(line, stdout); if (strstr(line, "GT sheet voxels")) sscanf(line, "gt-gate: %lld GT sheet voxels, %lld cells with >= %*d (smaller cells kept unchecked), %lld gated (support < %*f) = %lf", &sh, &cells, &gc, &share); }
    if (f) fclose(f);
    CHECK(sh == nsheet, "sheet voxels %lld (fixture %ld)", sh, nsheet);
    CHECK(cells == 8 && gc == 4, "cells counted %lld, gated %lld (want 8 / 4: the x >= 64 half)", cells, gc);
    CHECK(fabs(share - (double)nright / nsheet) < 0.02, "gated share %.3f (the x >= 64 sheet share %.3f)", share, (double)nright / nsheet);
    char key[64]; z3w_level_name(1, key, sizeof key);
    store *s = store_open(q); z3 *z = s ? z3_open(s, key, nullptr) : nullptr; CHECK(z != nullptr, "the gate mask %s/%s", q, key);
    if (z) {
        uint8_t *m = malloc((size_t)N * N * N); const int64_t o[3] = {0, 0, 0}, n[3] = {N, N, N};
        CHECK(!z3_read(z, o, n, m, 2), "read the mask");
        long bad = 0; for (int zz = 0; zz < N; zz += 7) for (int y = 0; y < N; y += 7) for (int x = 0; x < N; x += 7) bad += m[((size_t)zz * N + y) * N + x] != (x >= 64 ? 255 : 0);
        CHECK(bad == 0, "the mask is 255 on the x >= 64 cells, 0 elsewhere (%ld samples off)", bad);
        free(m); z3_close(z);
    }
    snprintf(p, sizeof p, "rm -rf -- '%s'", D); if (!fails && system(p)) {}
    free(gt); free(pb);
    printf(fails ? "test_gt_gate: %d failure(s)\n" : "test_gt_gate: ok\n", fails);
    return fails != 0;
}
