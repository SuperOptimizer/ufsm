/* ufsm fields-pack (src/fieldspack.c): a zarr fit-field set re-encoded
   - --q-phase 0: every plane's values unchanged (phase lossless, the others' chunks copied), 512^3 shards, loadable by fit_fields_load;
   - the default (--q-phase 4, predict's): the phase planes lossy, the confidence-weighted phase error small, the other planes unchanged;
   - --mask-w 1: pc = ps = 128 where w = 0, unchanged elsewhere;
   - a crop (fields.json zarr_offset): only the shards meeting the box written, unchanged arrays linked. */
#include "fit.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_fields_pack(int argc, char **argv);   /* src/fieldspack.c */

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 40, H = 150, W = 140, NP = 11 };   /* (H > 128: two shards in y at the input's 128^3 shards) */
static const char *nm[NP] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "r"};
static uint8_t V[NP][D * H * W];

static void mkset(const char *dir, const char *extra_json) {
    mkdir(dir, 0755);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double r = hypot(y - 70.0, x - 66.0), phi = r / 6.5 + 0.01 * z;
        const int mat = r > 8 && r < 60 && (int)(r / 3) % 4 != 0;
        const double conf = mat ? 0.55 + 0.4 * sin(0.05 * x + 0.03 * y) * sin(0.07 * z) : 0.0;
        V[9][i] = mat ? (uint8_t)(120 + 100 * sin(0.2 * r)) : (r < 60 && (x + y) % 7 == 0 ? 1 : 0);
        if (!mat && r < 60 && (x + y + z) % 5 == 0) V[9][i] = 0;
        const double c = mat || r < 60 ? (mat ? conf : 0.05) : 0;   /* weak phase in the gaps (w often 0 there) */
        V[0][i] = (uint8_t)lround(128 + 127 * c * cos(2 * M_PI * phi)); V[1][i] = (uint8_t)lround(128 + 127 * c * sin(2 * M_PI * phi));
        const double ny = (y - 70.0) / (r + 1e-9), nx = (x - 66.0) / (r + 1e-9), nz = 0.1, nn = sqrt(ny * ny + nx * nx + nz * nz);
        const double n3[3] = {nz / nn, ny / nn, nx / nn};
        for (int k = 0; k < 3; k++) V[2 + k][i] = mat ? (uint8_t)lround(255 * n3[k] * n3[k]) : 0;
        V[5][i] = mat ? (uint8_t)lround(255 * (0.5 + n3[0] * n3[1])) : 0; V[6][i] = mat ? (uint8_t)lround(255 * (0.5 + n3[0] * n3[2])) : 0;
        V[7][i] = mat ? (uint8_t)lround(255 * (0.5 + n3[1] * n3[2])) : 0;
        V[8][i] = mat ? 52 : 0; V[10][i] = mat ? (uint8_t)(255 * fabs(cos(M_PI * phi))) : 0;
    }
    char p[600]; const int64_t sh[3] = {D, H, W}; uint8_t *sb = malloc(128 * 128 * 128);
    for (int k = 0; k < NP; k++) {
        snprintf(p, sizeof p, "%s/%s", dir, nm[k]);
        z3w *z = z3w_create(p, sh, 128, k < 2 || k == 10 ? 0 : 8, k < 2 ? 128 : 0, "{\"ufsm\":{\"h\":2}}"); CHECK(z != nullptr, "z3w %s", nm[k]);
        for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
            memset(sb, k < 2 ? 128 : 0, 128 * 128 * 128);
            for (int a = 0; a < D; a++) for (int b = 0; b < 128 && sy * 128 + b < H; b++) for (int c = 0; c < 128 && sx * 128 + c < W; c++)
                sb[((size_t)a * 128 + b) * 128 + c] = V[k][((size_t)a * H + sy * 128 + b) * W + sx * 128 + c];
            CHECK(z3w_write_shard(z, 0, sy, sx, sb, 1) == 0, "shard %s", nm[k]);
        }
        z3w_close(z);
    }
    free(sb);
    snprintf(p, sizeof p, "%s/fields.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"format\": \"zarr\", \"origin\": [0, 0, 0], \"shape\": [%d, %d, %d], \"h\": 2, \"um\": 9.6, \"recto\": true%s, \"axis\": [", D, H, W, extra_json);
    for (int z = 0; z < 2 * D + 1; z++) fprintf(f, "%s[140.0, 132.0]", z ? ", " : "");
    fprintf(f, "]}\n"); fclose(f);
}
static int pack(const char *in, const char *out, const char *a1, const char *v1, const char *a2, const char *v2) {
    char *av[10] = {"ufsm", "fields-pack", (char *)in, (char *)out, "--threads", "3"}; int ac = 6;
    if (a1) { av[ac++] = (char *)a1; av[ac++] = (char *)v1; }
    if (a2) { av[ac++] = (char *)a2; av[ac++] = (char *)v2; }
    return cmd_fields_pack(ac, av);
}
/* the array decoded over [0, D) x [0, H) x [0, W) */
static int rd(const char *dir, const char *plane, uint8_t *out) {
    store *s = store_open(dir); z3 *z = s ? z3_open(s, plane, nullptr) : nullptr; const int64_t o[3] = {0, 0, 0}, n[3] = {D, H, W};
    const int rc = z ? z3_read(z, o, n, out, 2) : -1; if (z) z3_close(z); if (s) store_close(s); return rc;
}
static int shard512(const char *dir, const char *plane) {
    char p[600]; snprintf(p, sizeof p, "%s/%s/zarr.json", dir, plane); FILE *f = fopen(p, "r"); if (!f) return 0;
    char b[4096]; const size_t n = fread(b, 1, sizeof b - 1, f); b[n] = 0; fclose(f); return strstr(b, "\"chunk_shape\":[512,512,512]") != nullptr;
}

int main(void) {
    char tmp[] = "/tmp/test_fields_pack_XXXXXX"; if (!mkdtemp(tmp)) { fprintf(stderr, "mkdtemp\n"); return 1; }
    char in[300], o1[300], o2[300], o3[300], cr[300], o4[300], p[600];
    snprintf(in, sizeof in, "%s/in", tmp); snprintf(o1, sizeof o1, "%s/o1", tmp); snprintf(o2, sizeof o2, "%s/o2", tmp);
    snprintf(o3, sizeof o3, "%s/o3", tmp); snprintf(cr, sizeof cr, "%s/crop", tmp); snprintf(o4, sizeof o4, "%s/o4", tmp);
    mkset(in, "");
    static uint8_t a[D * H * W], b[D * H * W], c[D * H * W], w[D * H * W];
    CHECK(rd(in, "w", w) == 0, "read w");

    /* --q-phase 0: values unchanged, 512 shards, loadable */
    CHECK(pack(in, o1, "--q-phase", "0", nullptr, nullptr) == 0, "pack --q-phase 0");
    for (int k = 0; k < NP; k++) {
        CHECK(rd(in, nm[k], a) == 0 && rd(o1, nm[k], b) == 0, "read %s", nm[k]);
        CHECK(!memcmp(a, b, sizeof a), "--q-phase 0: plane %s changed", nm[k]);
        CHECK(shard512(o1, nm[k]), "--q-phase 0: %s not in 512^3 shards", nm[k]);
    }
    fit_fields_buf g;
    CHECK(fit_fields_load(o1, 0, 0, 1, &g) == 0, "load the packed set");
    CHECK(g.shape[0] == D && g.shape[1] == H && g.shape[2] == W && g.f.recto != nullptr, "packed set geometry / recto plane");
    fit_fields_free(&g);

    /* the default (lossy phase, q 4): confidence-weighted error small; the other planes unchanged */
    CHECK(pack(in, o2, nullptr, nullptr, nullptr, nullptr) == 0, "pack (defaults)");
    CHECK(rd(in, "pc", a) == 0 && rd(in, "ps", b) == 0, "read in phase");
    {
        static uint8_t qa[D * H * W], qb[D * H * W]; CHECK(rd(o2, "pc", qa) == 0 && rd(o2, "ps", qb) == 0, "read q phase");
        double se = 0, sw = 0, emax = 0, ce = 0; int nchg = 0;
        for (size_t i = 0; i < sizeof a; i++) {
            if (!w[i]) continue;
            const double x0 = (a[i] - 128) / 127.0, y0 = (b[i] - 128) / 127.0, x1 = (qa[i] - 128) / 127.0, y1 = (qb[i] - 128) / 127.0, c0 = hypot(x0, y0);
            if (c0 < 0.3) continue;
            double e = (atan2(y1, x1) - atan2(y0, x0)) / (2 * M_PI); e -= round(e);
            se += c0 * fabs(e); sw += c0; emax = fmax(emax, fabs(e)); ce += fabs(hypot(x1, y1) - c0); nchg += qa[i] != a[i] || qb[i] != b[i];
        }
        CHECK(nchg > 0, "--q-phase 4 left the phase unchanged");
        CHECK(se / sw < 0.01 && emax < 0.15, "--q-phase 4: weighted mean phase error %.4f turn, max %.4f", se / sw, emax);
        fprintf(stderr, "test_fields_pack: --q-phase 4: mean |phase error| %.4f turn (conf-weighted), max %.4f, mean |conf error| %.4f\n", se / sw, emax, ce / sw);
    }
    for (int k = 2; k < NP; k++) { CHECK(rd(in, nm[k], a) == 0 && rd(o2, nm[k], b) == 0, "read %s", nm[k]); CHECK(!memcmp(a, b, sizeof a), "--q-phase: plane %s changed", nm[k]); }
    snprintf(p, sizeof p, "%s/pc/zarr.json", o2); { FILE *f = fopen(p, "r"); char t[4096] = {0}; if (f) { if (!fread(t, 1, sizeof t - 1, f)) t[0] = 0; fclose(f); }
      CHECK(strstr(t, "\"q\":4") && strstr(t, "\"fields_pack\"") && strstr(t, "\"ufsm\":{\"h\":2}"), "pc zarr.json: codec q / attributes: %s", t); }

    /* a store already at the target quantiser: copied, not re-encoded (no second generation loss) */
    { char o5[300]; snprintf(o5, sizeof o5, "%s/o5", tmp);
      CHECK(pack(o2, o5, "--q-phase", "4", nullptr, nullptr) == 0, "re-pack at the same q");
      CHECK(rd(o2, "pc", a) == 0 && rd(o5, "pc", b) == 0 && !memcmp(a, b, sizeof a), "re-pack at the same q changed pc"); }

    /* --mask-w 1 */
    CHECK(pack(in, o3, "--mask-w", "1", "--q-phase", "0") == 0, "pack --mask-w 1");
    CHECK(rd(in, "pc", a) == 0 && rd(o3, "pc", b) == 0 && rd(o3, "ps", c) == 0, "read masked");
    { int bad = 0, nm0 = 0; for (size_t i = 0; i < sizeof a; i++) { if (!w[i]) { bad += b[i] != 128 || c[i] != 128; nm0 += a[i] != 128; } else bad += b[i] != a[i]; }
      CHECK(!bad && nm0 > 0, "--mask-w: %d cells wrong (%d input cells had phase at w = 0)", bad, nm0); }

    /* a crop (zarr_offset) of the same arrays: only shards meeting the box; unchanged planes linked */
    mkdir(cr, 0755);
    for (int k = 0; k < NP; k++) { char t[600]; snprintf(p, sizeof p, "%s/%s", in, nm[k]); snprintf(t, sizeof t, "%s/%s", cr, nm[k]); CHECK(symlink(p, t) == 0, "link"); }
    snprintf(p, sizeof p, "%s/fields.json", cr); { FILE *f = fopen(p, "w");
      fprintf(f, "{\"format\": \"zarr\", \"origin\": [0, 0, 0], \"shape\": [8, 8, 8], \"h\": 2, \"um\": 9.6, \"zarr_offset\": [0, 0, 0], \"axis\": [");
      for (int z = 0; z < 17; z++) fprintf(f, "%s[140.0, 132.0]", z ? ", " : "");
      fprintf(f, "]}\n"); fclose(f); }
    CHECK(pack(cr, o4, "--q-phase", "4", nullptr, nullptr) == 0, "pack a crop");
    { struct stat sb; char t[600]; snprintf(t, sizeof t, "%s/t0", o4);
      CHECK(lstat(t, &sb) == 0 && S_ISLNK(sb.st_mode), "crop: unchanged plane t0 not linked");
      snprintf(t, sizeof t, "%s/pc/c/0/0/0", o4); CHECK(stat(t, &sb) == 0, "crop: the box's pc shard missing"); }
    CHECK(fit_fields_load(o4, 0, 0, 1, &g) == 0 && g.shape[0] == 8, "load the packed crop");
    if (!fails) fit_fields_free(&g);

    snprintf(p, sizeof p, "rm -rf %s", tmp); if (system(p)) fprintf(stderr, "cleanup failed\n");
    if (fails) { fprintf(stderr, "test_fields_pack: %d FAILED\n", fails); return 1; }
    printf("test_fields_pack: OK\n");
    return 0;
}
