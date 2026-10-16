/* Side votes (src/votes.h, src/votes.c, sources "encoding": "side-votes"): the diagonal code is a bijection of the 16 x 16 (r, v)
   grid whose single votes are the side codes (RV_VERSO 1, RV_RECTO 2); adding codes adds counts (saturating); the decode rules;
   side_votes_merge of two side arrays and of a vote array with a side array on a synthetic grid (edge shards, an input absent from a
   shard); the sampler's reads of the merged array under each rule, and a side array read as "side" vs as "side-votes". */
#include "votes.h"
#include "band.h"
#include "sources.h"
#include "zarr3.h"
#include "z3w.h"
#include "store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { NZ = 200, NY = 256, NX = 320, SH = 128 };
static uint8_t side_a(int z, int y, int x) { (void)z; return y < 200 ? ((x / 10) % 2 ? RV_VERSO : RV_RECTO) : RV_UNKNOWN; }      /* set A */
static uint8_t side_b(int z, int y, int x) { return y >= 50 && z < 150 ? (((x + 2) / 10) % 2 ? RV_VERSO : RV_RECTO) : RV_UNKNOWN; }   /* B: shifted, partial */

static void write_array(const char *group, uint8_t (*f)(int, int, int)) {
    char d[1200]; snprintf(d, sizeof d, "%s/4.8", group);
    const int64_t shape[3] = {NZ, NY, NX};
    z3w *w = z3w_create(d, shape, SH, 0, 0, "{\"ufsm\":{\"encoding\":\"side\"}}");
    uint8_t *buf = malloc((size_t)SH * SH * SH);
    for (int sz = 0; sz * SH < NZ; sz++) for (int sy = 0; sy * SH < NY; sy++) for (int sx = 0; sx * SH < NX; sx++) {
        for (int z = 0; z < SH; z++) for (int y = 0; y < SH; y++) for (int x = 0; x < SH; x++) {
            const int gz = sz * SH + z, gy = sy * SH + y, gx = sx * SH + x;
            buf[((size_t)z * SH + y) * SH + x] = gz < NZ && gy < NY && gx < NX ? f(gz, gy, gx) : 0;
        }
        CHECK(!z3w_write_shard(w, sz, sy, sx, buf, 2), "write %s: %s", group, z3w_error());
    }
    z3w_close(w); free(buf);
    const double um = 4.8; CHECK(!z3w_write_group(group, &um, 1, "test", nullptr), "group %s", group);
}

static uint8_t *read_all(const char *root, const char *key) {
    z3 *z = z3_open(store_open(root), key, nullptr);
    if (!z) { fprintf(stderr, "open %s/%s: %s\n", root, key, z3_error()); fails++; return calloc((size_t)NZ * NY * NX, 1); }
    uint8_t *b = malloc((size_t)NZ * NY * NX); const int64_t o[3] = {0, 0, 0}, n[3] = {NZ, NY, NX};
    CHECK(!z3_read(z, o, n, b, 2), "read %s: %s", key, z3_error());
    z3_close(z); return b;
}

int main(void) {
    /* the code */
    int seen[256] = {0};
    for (int r = 0; r <= VOTE_MAX; r++) for (int v = 0; v <= VOTE_MAX; v++) {
        const uint8_t c = vote_code(r, v); int r2, v2; vote_split(c, &r2, &v2);
        seen[c]++; CHECK(r2 == r && v2 == v, "(%d,%d) -> %d -> (%d,%d)", r, v, c, r2, v2);
    }
    for (int c = 0; c < 256; c++) CHECK(seen[c] == 1, "code %d used %d times", c, seen[c]);
    CHECK(vote_code(0, 0) == RV_UNKNOWN && vote_code(0, 1) == RV_VERSO && vote_code(1, 0) == RV_RECTO, "single votes are the side codes");
    CHECK(vote_code(0, 2) == 3 && vote_code(1, 1) == 4 && vote_code(2, 0) == 5 && vote_code(15, 15) == 255, "diagonal order");
    CHECK(vote_add(vote_code(9, 3), vote_code(9, 1)) == vote_code(15, 4) && vote_add(RV_RECTO, RV_VERSO) == vote_code(1, 1), "vote_add");
    uint8_t lu[256], lm[256], lu2[256];
    vote_side_lut(VOTE_UNANIMOUS, 1, lu); vote_side_lut(VOTE_MAJORITY, 1, lm); vote_side_lut(VOTE_UNANIMOUS, 2, lu2);
    CHECK(lu[RV_RECTO] == RV_RECTO && lu[RV_VERSO] == RV_VERSO && lu[0] == 0 && lu[vote_code(1, 1)] == 0 && lu[vote_code(3, 1)] == 0, "unanimous");
    CHECK(lm[vote_code(3, 1)] == RV_RECTO && lm[vote_code(1, 3)] == RV_VERSO && lm[vote_code(2, 2)] == 0, "majority");
    CHECK(lu2[RV_RECTO] == 0 && lu2[vote_code(2, 0)] == RV_RECTO && lu2[vote_code(0, 2)] == RV_VERSO, "min 2");
    for (int c = 0; c < 3; c++) CHECK(lu[c] == c && lm[c] == c, "one set's side codes decode to themselves (%d)", c);

    /* merges on a synthetic grid (inputs of 128^3 shards, output shards of 256: edge shards, B absent where z >= 150 ... ) */
    char root[] = "/tmp/ufsm-votes-test-XXXXXX";
    if (!mkdtemp(root)) return 2;
    char ga[1300], gb[1300], gm[1300], gm2[1300];
    snprintf(ga, sizeof ga, "%s/a", root); snprintf(gb, sizeof gb, "%s/b", root); snprintf(gm, sizeof gm, "%s/m", root); snprintf(gm2, sizeof gm2, "%s/m2", root);
    write_array(ga, side_a); write_array(gb, side_b);
    store *st = store_open(root);
    z3 *in[2] = {z3_open(st, "a/4.8", nullptr), z3_open(st, "b/4.8", nullptr)};
    const char *names[2] = {"a", "b"}; size_t stats[2]; int nsh = 0;
    CHECK(in[0] && in[1] && !side_votes_merge(gm, 4.8, in, names, 2, 256, 2, &nsh, stats), "merge a + b: %s", side_votes_error());
    uint8_t *M = read_all(root, "m/4.8");
    size_t bad = 0, conf = 0, nvote = 0;
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const uint8_t a = side_a(z, y, x), b = side_b(z, y, x);
        const int r = (a == RV_RECTO) + (b == RV_RECTO), v = (a == RV_VERSO) + (b == RV_VERSO);
        bad += M[((size_t)z * NY + y) * NX + x] != vote_code(r, v); conf += r && v; nvote += r + v > 0;
    }
    CHECK(!bad, "merge a + b: %zu voxels differ from the counts", bad);
    CHECK(stats[0] == nvote && stats[1] == conf, "merge stats %zu / %zu vs %zu / %zu", stats[0], stats[1], nvote, conf);
    printf("merge a + b: %d shards, %zu voxels with votes, %zu disputed (%s)\n", nsh, nvote, conf, bad ? "FAIL" : "exact");
    z3 *in2[2] = {z3_open(st, "m/4.8", nullptr), z3_open(st, "a/4.8", nullptr)};   /* a vote array plus a side array */
    const char *names2[2] = {"m", "a"};
    CHECK(in2[0] && in2[1] && !side_votes_merge(gm2, 4.8, in2, names2, 2, 128, 2, &nsh, stats), "merge m + a: %s", side_votes_error());
    uint8_t *M2 = read_all(root, "m2/4.8"); bad = 0;
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const uint8_t a = side_a(z, y, x), b = side_b(z, y, x);
        bad += M2[((size_t)z * NY + y) * NX + x] != vote_code(2 * (a == RV_RECTO) + (b == RV_RECTO), 2 * (a == RV_VERSO) + (b == RV_VERSO));
    }
    CHECK(!bad, "merge (a + b) + a: %zu voxels differ", bad);

    /* the sampler's reads: a side array as "side" and as "side-votes"; the merged array under each rule */
    char sj[1400]; snprintf(sj, sizeof sj, "%s/sources.json", root);
    FILE *f = fopen(sj, "w");
    fprintf(f, "{\"sources\": [\n"
               " {\"name\": \"side\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"a\", \"encoding\": \"side\"}}},\n"
               " {\"name\": \"a-votes\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"a\", \"encoding\": \"side-votes\"}}},\n"
               " {\"name\": \"unan\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"m2\", \"encoding\": \"side-votes\"}}},\n"
               " {\"name\": \"maj\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"m2\", \"encoding\": \"side-votes\", \"votes\": {\"rule\": \"majority\"}}}},\n"
               " {\"name\": \"min3\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"m2\", \"encoding\": \"side-votes\", \"votes\": {\"min\": 3}}}},\n"
               " {\"name\": \"soft\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 4.8, \"iso\": 60, \"targets\": {\"recto\": {\"group\": \"m2\", \"encoding\": \"side-votes\", \"votes\": {\"soft\": 1}}}}\n"
               "]}\n", root, root, root, root, root, root);
    fclose(f);
    sources *S = sources_load(sj);
    CHECK(S && S->n == 6, "sources_load");
    if (S && S->n == 6) {
        const int64_t o[3] = {10, 30, 5}, n[3] = {150, 200, 300}; const size_t nv = (size_t)n[0] * n[1] * n[2];
        uint8_t *rd[6];
        for (int i = 0; i < 6; i++) { rd[i] = malloc(nv); CHECK(!source_read_target(&S->src[i], CH_RECTO, 0, o, n, rd[i], 2), "read %s", S->src[i].name); }
        CHECK(!memcmp(rd[0], rd[1], nv), "a side array reads the same as \"side\" and as \"side-votes\"");
        CHECK(S->src[1].tgt_side[CH_RECTO] && S->src[1].tgt_votes[CH_RECTO] && !S->src[0].tgt_votes[CH_RECTO], "side-votes sets tgt_side");
        size_t e[4] = {0}, kn[4] = {0};
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
            const size_t k = ((size_t)z * n[1] + y) * n[2] + x;
            const uint8_t c = M2[((size_t)(o[0] + z) * NY + o[1] + y) * NX + o[2] + x]; int r, v; vote_split(c, &r, &v);
            const uint8_t un = r && !v ? RV_RECTO : v && !r ? RV_VERSO : 0, mj = r > v ? RV_RECTO : v > r ? RV_VERSO : 0, m3 = r >= 3 && !v ? RV_RECTO : v >= 3 && !r ? RV_VERSO : 0;
            e[0] += rd[2][k] != un; e[1] += rd[3][k] != mj; e[2] += rd[4][k] != m3; e[3] += rd[5][k] != c;
            kn[0] += un != 0; kn[1] += mj != 0; kn[2] += m3 != 0; kn[3] += S->src[5].tgt_vote_lut[CH_RECTO][c] != 255;
        }
        CHECK(!e[0] && !e[1] && !e[2] && !e[3], "rule reads differ: unanimous %zu, majority %zu, min 3 %zu, soft (raw codes) %zu", e[0], e[1], e[2], e[3]);
        CHECK(S->src[5].tgt_votes_soft[CH_RECTO] && S->src[5].tgt_vote_lut[CH_RECTO][vote_code(2, 1)] == 169 && S->src[5].tgt_vote_lut[CH_RECTO][RV_VERSO] == 0 &&
              S->src[5].tgt_vote_lut[CH_RECTO][RV_RECTO] == 254 && S->src[5].tgt_vote_lut[CH_RECTO][0] == 255, "soft table");
        printf("reads of (a + b) + a: labelled unanimous %zu, majority %zu, min 3 %zu, soft %zu of %zu voxels\n", kn[0], kn[1], kn[2], kn[3], nv);
        for (int i = 0; i < 6; i++) free(rd[i]);
        sources_free(S);
    }
    free(M); free(M2);
    printf("side votes: %s (%s)\n", fails ? "FAIL" : "PASS", root);
    if (getenv("UFSM_VOTES_KEEP")) return fails != 0;
    { char c[1400]; snprintf(c, sizeof c, "rm -rf %s", root); if (system(c)) {} }
    return fails != 0;
}
