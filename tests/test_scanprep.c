/* ufsm scanprep (src/scanprep.c) on a synthetic masked scroll, and the sampler's unlabelled-scan support (sources "mask", "ct_only",
   "norm"): the building blocks (Otsu, closing, hole fill, valid centres, histogram quantiles) on small grids; then a 256^3 volume
   (levels 0 and 2, integer level names) of a papyrus ring (CT ~160) around an air core (~60), air gaps, zero outside a masking
   cylinder: the mask must hold the ring and the filled core, exclude the masked-out corners, the statistics must see both papyrus and
   air; sampler windows of a ct_only source must centre on valid cells, carry no targets and use the fixed normalisation. */
#include "../src/sample.c"   /* (the sampler compiled in, no CUDA objects: as test_cond / test_sampler_safety) */
#include "../src/scanprep.h"
#include "../src/z3w.h"
#include "../src/json.h"
void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static uint64_t rs = 12345;
static double urand(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (double)(rs >> 11) / 9007199254740992.0; }

static void write_level(const char *dir, const uint8_t *v, int N) {
    const int64_t sh[3] = {N, N, N}; const int shard = 128; z3w *w = z3w_create(dir, sh, shard, 0, 0, nullptr);
    CHECK(w != nullptr, "z3w_create %s", dir); if (!w) return;
    uint8_t *b = calloc((size_t)shard * shard * shard, 1); const int ns = (N + shard - 1) / shard;
    for (int sz = 0; sz < ns; sz++) for (int sy = 0; sy < ns; sy++) for (int sx = 0; sx < ns; sx++) {
        memset(b, 0, (size_t)shard * shard * shard);
        for (int z = 0; z < shard && sz * shard + z < N; z++) for (int y = 0; y < shard && sy * shard + y < N; y++) for (int x = 0; x < shard && sx * shard + x < N; x++)
            b[((size_t)z * shard + y) * shard + x] = v[((size_t)(sz * shard + z) * N + sy * shard + y) * N + sx * shard + x];
        CHECK(!z3w_write_shard(w, sz, sy, sx, b, 2), "write shard");
    }
    CHECK(!z3w_close(w), "close"); free(b);
}

static void test_blocks(void) {
    double h[256] = {0}; for (int v = 50; v < 70; v++) h[v] = 100; for (int v = 150; v < 170; v++) h[v] = 50;
    const int t = sp_otsu(h, 256); CHECK(t > 69 && t <= 150, "otsu %d between the modes", t);
    double h1[256] = {0}; h1[7] = 5; CHECK(sp_otsu(h1, 256) == -1, "otsu of one bin");
    uint64_t q[256] = {0}; q[10] = 1; q[20] = 2; q[30] = 1;
    CHECK(sp_hist_quantile(q, 0.5) == 20 && sp_hist_quantile(q, 0.0) == 10 && sp_hist_quantile(q, 1.0) == 30, "quantiles");
    /* closing joins two slabs 2 cells apart (r 2), leaves a lone far cell, does not erode at the grid border */
    enum { N = 20 }; const int64_t n[3] = {N, N, N}; uint8_t *m = calloc(N * N * N, 1);
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) m[(z * N + y) * N + x] = x < 8 || (x >= 10 && x < 16);
    CHECK(!sp_close(m, n, 2.f, 2), "close");
    int gap = 0, border = 0; for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) { gap += m[(z * N + y) * N + 9]; border += m[(z * N + y) * N + 0]; }
    CHECK(gap == N * N && border == N * N, "closing: gap %d border %d of %d", gap, border, N * N);
    CHECK(m[(10 * N + 10) * N + 19] == 0, "closing keeps background past the slab");
    /* hole fill: a closed hollow box fills; a tube open in z fills only with the 2-D pass */
    memset(m, 0, N * N * N);
    for (int z = 2; z < 18; z++) for (int y = 2; y < 18; y++) for (int x = 2; x < 18; x++) m[(z * N + y) * N + x] = z == 2 || z == 17 || y == 2 || y == 17 || x == 2 || x == 17;
    CHECK(!sp_fill_holes(m, n, 0), "fill"); CHECK(m[(10 * N + 10) * N + 10] == 1 && m[0] == 0, "box filled, outside not");
    memset(m, 0, N * N * N);
    for (int z = 0; z < N; z++) for (int y = 2; y < 18; y++) for (int x = 2; x < 18; x++) m[(z * N + y) * N + x] = y == 2 || y == 17 || x == 2 || x == 17;
    CHECK(!sp_fill_holes(m, n, 0), "fill 3d"); CHECK(m[(10 * N + 10) * N + 10] == 0, "open tube: 3-D fill leaves the core");
    CHECK(!sp_fill_holes(m, n, 1), "fill 2d"); CHECK(m[(10 * N + 10) * N + 10] == 1 && m[(10 * N + 0) * N + 0] == 0, "open tube: 2-D fill fills the core");
    /* components: a 10^3 block and a 2^3 speck; frac 0.1 keeps the block only, 0.001 both */
    for (int pass = 0; pass < 2; pass++) {
        memset(m, 0, N * N * N);
        for (int z = 0; z < 10; z++) for (int y = 0; y < 10; y++) for (int x = 0; x < 10; x++) m[(z * N + y) * N + x] = 1;
        for (int z = 15; z < 17; z++) for (int y = 15; y < 17; y++) for (int x = 15; x < 17; x++) m[(z * N + y) * N + x] = 1;
        int64_t nc = 0; const int64_t kept = sp_keep_components(m, n, pass ? 0.001 : 0.1, &nc);
        int tot = 0; for (int k = 0; k < N * N * N; k++) tot += m[k];
        CHECK(nc == 2 && kept == (pass ? 2 : 1) && tot == (pass ? 1008 : 1000), "components pass %d: %lld found, %lld kept, %d cells", pass, (long long)nc, (long long)kept, tot);
    }
    /* valid centres: a full grid is valid everywhere with min_inside up to 1/8 at a corner */
    memset(m, 1, N * N * N); uint8_t *o = malloc(N * N * N);
    CHECK(sp_valid_centres(m, n, 4, 0.125, o) == N * N * N, "all valid at 1/8");
    CHECK(sp_valid_centres(m, n, 4, 1.0, o) == (int64_t)(N - 3) * (N - 3) * (N - 3), "full boxes only inside: %lld", (long long)sp_valid_centres(m, n, 4, 1.0, o));
    free(m); free(o);
}

static void test_volume(const char *tmp) {
    enum { N = 256 }; const size_t n3 = (size_t)N * N * N;
    uint8_t *v = malloc(n3), *v2 = malloc(n3 / 64);
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const double r = hypot(y - 127.5, x - 127.5); uint8_t c;
        if (r > 100) c = 0;                                       /* the scan's mask */
        else if (r < 30) c = (uint8_t)(60 + 8 * (urand() - 0.5));   /* air core */
        else if (fmod(r, 6.0) < 1.0) c = (uint8_t)(60 + 8 * (urand() - 0.5));   /* air gaps between wraps */
        else c = (uint8_t)(160 + 20 * (urand() - 0.5));          /* papyrus */
        if (r > 92) c = c ? (uint8_t)(60 + 8 * (urand() - 0.5)) : 0;   /* air between the scroll and the mask edge */
        v[((size_t)z * N + y) * N + x] = c;
    }
    for (int z = 0; z < N / 4; z++) for (int y = 0; y < N / 4; y++) for (int x = 0; x < N / 4; x++) {   /* level 2: 4^3 means (with zeros) */
        double s = 0; for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) for (int c = 0; c < 4; c++) s += v[((size_t)(4 * z + a) * N + 4 * y + b) * N + 4 * x + c];
        v2[((size_t)z * (N / 4) + y) * (N / 4) + x] = (uint8_t)lround(s / 64);
    }
    char d[700]; snprintf(d, sizeof d, "%s/vol.zarr/0", tmp); write_level(d, v, N);
    snprintf(d, sizeof d, "%s/vol.zarr/2", tmp); write_level(d, v2, N / 4);
    char out[700]; snprintf(out, sizeof out, "%s/prep", tmp);
    char *argv[] = {"ufsm", "scanprep", (char *)tmp, "vol.zarr", "--level", "0", "--um", "8", "--cell", "16", "--close-mm", "0.3", "--window", "64",
                    "--min-inside", "0.25", "--stats-chunks", "16", "--threads", "2", "--out", out};
    CHECK(cmd_scanprep((int)(sizeof argv / sizeof *argv), argv) == 0, "scanprep");
    char p[800]; snprintf(p, sizeof p, "%s.mask.raw", out);
    FILE *f = fopen(p, "rb"); uint8_t mk[16 * 16 * 16]; CHECK(f && fread(mk, 1, sizeof mk, f) == sizeof mk, "mask raw 16^3"); if (f) fclose(f);
    int corner_bad = 0, ring_bad = 0, core_bad = 0, nvalid = 0;
    for (int z = 0; z < 16; z++) for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
        const double r = hypot(16 * y + 7.5 - 127.5, 16 * x + 7.5 - 127.5); const uint8_t c = mk[(z * 16 + y) * 16 + x];
        if (r > 115) corner_bad += c != 0;          /* every voxel of the cell is masked out */
        if (r > 40 && r < 85) ring_bad += c == 0;    /* papyrus */
        if (r < 20) core_bad += c == 0;              /* the air core, filled per plane */
        nvalid += c == 2;
    }
    CHECK(!corner_bad && !ring_bad && !core_bad && nvalid > 0, "mask: %d masked-out cells kept, %d ring / %d core cells lost, %d valid", corner_bad, ring_bad, core_bad, nvalid);
    snprintf(p, sizeof p, "%s.json", out);
    f = fopen(p, "rb"); char jb[16384] = {0}; size_t jn = f ? fread(jb, 1, sizeof jb - 1, f) : 0; if (f) fclose(f);
    json *j = jn ? json_parse(jb, jn) : nullptr; CHECK(j != nullptr, "prep json parses");
    const double lo = json_num(json_path(j, "stats.p0_5"), -1), hi = json_num(json_path(j, "stats.p99_5"), -1), med = json_num(json_path(j, "stats.median"), -1),
                 sc = json_num(json_path(j, "stats.scale"), -1), cut = json_num(json_path(j, "stats.otsu"), -1);
    CHECK(lo >= 55 && lo <= 65 && hi >= 165 && hi <= 171 && med >= 140 && med <= 171 && sc > 0 && cut > 70 && cut < 150,
          "stats p0.5 %.0f p99.5 %.0f median %.0f scale %.2f otsu %.0f", lo, hi, med, sc, cut);
    CHECK(json_num(json_get(j, "valid_cells"), -1) == nvalid && json_num(json_get(j, "lev"), -1) == 4, "json cells / lev");
    json_free(j);
    /* the sampler: a ct_only source with the mask and a fixed normalisation */
    char js[800]; snprintf(js, sizeof js, "%s/sources.json", tmp);
    f = fopen(js, "w");
    fprintf(f, "{\"sources\":[{\"name\":\"syn\",\"root\":\"%s\",\"ct\":\"vol.zarr\",\"um\":8,\"ct_only\":1,\"mask\":{\"map\":\"%s.mask.raw\"},\"norm\":{\"center\":%g,\"scale\":%g}}]}\n",
            tmp, out, med, sc);
    fclose(f);
    sources *S = sources_load(js); CHECK(S && S->n == 1 && S->src[0].mask && S->src[0].ct_only && S->src[0].norm_s == (float)sc, "sources: mask / ct_only / norm");
    if (!S) { free(v); free(v2); return; }
    enum { P = 32, B = 4 }; const size_t p3 = (size_t)P * P * P;
    sample_cfg c = sample_cfg_default(); c.P = P; c.B = B; c.nworkers = 2; c.nbuf = 2; c.seed = 7; c.augment = 0; c.snap = 0; c.min_fg = 0.1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    sampler *sp = sampler_start(S, &c); CHECK(sp != nullptr, "sampler_start");
    int off = 0, wbad = 0, xbad = 0, nbat = 0;
    for (int bi = 0; sp && bi < 8; bi++) {
        batch *b = sampler_next(sp); if (!b) { CHECK(0, "sampler stopped"); break; }
        nbat++;
        for (int i = 0; i < B; i++) {
            const int64_t *o = b->corner[i]; int cc[3]; for (int d = 0; d < 3; d++) cc[d] = (int)((o[d] + P / 2) / 16);
            off += mk[(cc[0] * 16 + cc[1]) * 16 + cc[2]] != 2;
            for (int ch = 0; ch < NCH; ch++) wbad += b->w[i * NCH + ch] != 0;
            for (int k = 0; k < 64; k++) {   /* the CT channel: (v - median) / scale exactly (no augmentation) */
                const int z = k % P, y = (k * 7) % P, x = (k * 13) % P; const size_t q = ((size_t)z * P + y) * P + x;
                const double want = ((double)v[((size_t)(o[0] + z) * N + o[1] + y) * N + o[2] + x] - med) / sc, got = b->x[(size_t)i * 4 * p3 + q];
                xbad += fabs(got - want) > 1e-3;
            }
        }
        sampler_release(sp, b);
    }
    CHECK(nbat == 8 && !off && !wbad && !xbad, "sampler: %d batches, %d windows off valid cells, %d target flags, %d CT values off the normalisation", nbat, off, wbad, xbad);
    if (sp) sampler_stop(sp);
    sources_free(S); free(v); free(v2);
}

int main(void) {
    char tmp[] = "/tmp/test_scanprep_XXXXXX"; if (!mkdtemp(tmp)) { perror("mkdtemp"); return 1; }
    store_global_init();
    test_blocks();
    test_volume(tmp);
    char cmd[200]; snprintf(cmd, sizeof cmd, "rm -rf %s", tmp); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_scanprep: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_scanprep: ok\n");
    return 0;
}
