/* fields-derive (src/fieldsderive.c): a planar phase field of known normal and period -> the derived t0..t5 encode n n^T and sp
   the period (16 pi / |grad phase|); cells without material stay 0; the other planes are links. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_fields_derive(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 40, H = 36, W = 44 };
static uint8_t *rd(const char *dir, const char *k) {
    store *s = store_open(dir); z3 *z = s ? z3_open(s, k, nullptr) : nullptr; if (!z) return nullptr;
    uint8_t *o = malloc((size_t)D * H * W); const int64_t org[3] = {0, 0, 0}, n[3] = {D, H, W};
    if (z3_read(z, org, n, o, 1)) { free(o); o = nullptr; }
    z3_close(z); return o;
}

int main(void) {
    char tmp[] = "/tmp/test_fieldsderive_XXXXXX"; if (!mkdtemp(tmp)) return 1;
    char in[600], out[600], p[700]; snprintf(in, sizeof in, "%s/in", tmp); snprintf(out, sizeof out, "%s/out", tmp); mkdir(in, 0755);
    const double n[3] = {0.6, 0.0, 0.8}, P = 16.0;   /* the sheet normal (z, y, x) and the period in cells */
    static const char *nm[14] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "dz", "dy", "dx", "dq"};
    const int64_t sh[3] = {D, H, W}; uint8_t *sb = calloc(128 * 128 * 128, 1);
    for (int k = 0; k < 14; k++) {
        snprintf(p, sizeof p, "%s/%s", in, nm[k]); z3w *z = z3w_create(p, sh, 128, 0, k < 2 ? 128 : 0, nullptr); CHECK(z, "z3w %s", nm[k]);
        memset(sb, k < 2 ? 128 : 0, 128 * 128 * 128);
        for (int a = 0; a < D; a++) for (int b = 0; b < H; b++) for (int c = 0; c < W; c++) {
            const double ph = 2 * M_PI * (n[0] * a + n[1] * b + n[2] * c) / P; const int mat = c >= 4;
            uint8_t v = 0;
            if (k == 0) v = mat ? (uint8_t)lround(128 + 127 * cos(ph)) : 128; else if (k == 1) v = mat ? (uint8_t)lround(128 + 127 * sin(ph)) : 128;
            else if (k == 9) v = mat ? 255 : 0; else if (k >= 2 && k <= 8) v = mat ? 77 : 0;   /* (the network's planes: replaced) */
            sb[((size_t)a * 128 + b) * 128 + c] = v;
        }
        CHECK(z3w_write_shard(z, 0, 0, 0, sb, 1) == 0, "shard %s", nm[k]); z3w_close(z);
    }
    free(sb);
    snprintf(p, sizeof p, "%s/fields.json", in); FILE *f = fopen(p, "w");
    fprintf(f, "{\"format\": \"zarr\", \"origin\": [0, 0, 0], \"shape\": [%d, %d, %d], \"h\": 2, \"um\": 9.6, \"axis\": [[0, 0]]}\n", D, H, W); fclose(f);

    char *av[] = {"ufsm", "fields-derive", in, out, "--r", "2", "--q", "0", "--threads", "2", nullptr};
    CHECK(cmd_fields_derive(10, av) == 0, "fields-derive");
    uint8_t *t[7]; static const char *on[7] = {"t0", "t1", "t2", "t3", "t4", "t5", "sp"};
    for (int k = 0; k < 7; k++) { t[k] = rd(out, on[k]); CHECK(t[k], "read %s", on[k]); if (!t[k]) return 1; }
    const double want[7] = {255 * n[0] * n[0], 255 * n[1] * n[1], 255 * n[2] * n[2], 255 * (0.5 + n[0] * n[1]), 255 * (0.5 + n[0] * n[2]), 255 * (0.5 + n[1] * n[2]), 16 * M_PI / (2 * M_PI / P)};
    double emax[7] = {0}; int nonmat_bad = 0;
    for (int a = 0; a < D; a++) for (int b = 0; b < H; b++) for (int c = 0; c < W; c++) {
        const size_t i = ((size_t)a * H + b) * W + c;
        if (c < 4) { for (int k = 0; k < 7; k++) nonmat_bad += t[k][i] != 0; continue; }
        if (a < 3 || a >= D - 3 || b < 3 || b >= H - 3 || c < 7 || c >= W - 3) continue;   /* (interior: the box and both differences inside) */
        for (int k = 0; k < 7; k++) { const double e = fabs(t[k][i] - want[k]); if (e > emax[k]) emax[k] = e; }
    }
    for (int k = 0; k < 7; k++) CHECK(emax[k] <= 3.0, "%s: max |error| %.1f vs %.1f", on[k], emax[k], want[k]);
    CHECK(nonmat_bad == 0, "%d plane values in cells without material", nonmat_bad);
    char lk[700]; snprintf(p, sizeof p, "%s/pc", out); const ssize_t L = readlink(p, lk, sizeof lk - 1); CHECK(L > 0, "pc is not a link");
    snprintf(p, sizeof p, "%s/t0", out); struct stat st; CHECK(lstat(p, &st) == 0 && !S_ISLNK(st.st_mode), "t0 should be written, not linked");
    for (int k = 0; k < 7; k++) free(t[k]);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", tmp); if (!strncmp(tmp, "/tmp/test_fieldsderive_", 23)) (void)!system(cmd);
    printf("%s\n", fails ? "test_fieldsderive: FAIL" : "test_fieldsderive: OK");
    return fails != 0;
}
