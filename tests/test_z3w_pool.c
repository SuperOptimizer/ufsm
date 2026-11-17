/* z3w_pool (2026-11-17, predict's heads: one pool of --threads encodes every head's shards chunk by chunk): the shard files it writes are
   z3w_write_shard's bytes, for three arrays at once (lossless, lossy q 8, a typed uint16 array) and shards with all-fill chunks and an
   all-fill shard (no object; an older object at its path removed); done runs once per shard; a failing shard (its directory a file)
   reports rc != 0 with the error text and the others are still written; destroy with shards queued finishes them. */
#include "z3w.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
enum { S = 256, NS = 6 };
static void fill(uint8_t *d, int es, int k) {   /* shard k: smooth field + noise, some chunks fill (k 3: the whole shard) */
    const size_t n = (size_t)S * S * S;
    for (size_t i = 0; i < n; i++) {
        const int z = (int)(i / (S * S)), y = (int)(i / S % S), x = (int)(i % S); const int ch = (z / 128) * 4 + (y / 128) * 2 + x / 128;
        unsigned v = (k == 3 || (ch + k) % 3 == 0) ? 0 : (unsigned)((z + 2 * y + 3 * x + 7 * k) & 255) ^ (xr() & (k & 1 ? 15 : 1));
        if (es == 1) d[i] = (uint8_t)v; else { const uint16_t w = v ? (uint16_t)(v * 251 + (xr() & 3)) : 0; memcpy(d + 2 * i, &w, 2); }
    }
}
static uint8_t *slurp(const char *p, long *n) { FILE *f = fopen(p, "rb"); if (!f) { *n = -1; return nullptr; } fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)*n + 1); if (fread(b, 1, (size_t)*n, f) != (size_t)*n) *n = -2; fclose(f); return b; }
typedef struct { atomic_int calls, bad; char err[256]; } dstat;
static void done(void *ctx, int rc, const char *err) { dstat *d = ctx; atomic_fetch_add(&d->calls, 1); if (rc) { atomic_fetch_add(&d->bad, 1); snprintf(d->err, sizeof d->err, "%s", err); } }
int main(void) {
    char tmp[400]; const char *td = getenv("TMPDIR"); snprintf(tmp, sizeof tmp, "%s/test_z3w_pool.XXXXXX", td && *td ? td : "/tmp"); if (!mkdtemp(tmp)) { perror("mkdtemp"); return 1; }
    const int64_t shape[3] = {(int64_t)NS * S, S, S};
    const char *nm[3] = {"q0", "q8", "u16"}; const int es[3] = {1, 1, 2};
    z3w *A[3], *B[3]; char d[512];
    for (int a = 0; a < 3; a++) for (int r = 0; r < 2; r++) {
        snprintf(d, sizeof d, "%s/%c%s", tmp, r ? 'b' : 'a', nm[a]);
        z3w *w = a == 2 ? z3w_create_typed(d, shape, S, 1, 0, 0, nullptr) : z3w_create(d, shape, S, a ? 8.f : 0.f, 0, nullptr);
        if (!w) { fprintf(stderr, "create %s: %s\n", d, z3w_error()); return 1; }
        (r ? B : A)[a] = w;
    }
    uint8_t *data[3][NS];
    for (int a = 0; a < 3; a++) for (int k = 0; k < NS; k++) { data[a][k] = malloc((size_t)S * S * S * es[a]); fill(data[a][k], es[a], k); }
    for (int a = 0; a < 3; a++) for (int k = 0; k < NS; k++) CHECK(!z3w_write_shard(A[a], k, 0, 0, data[a][k], 3), "z3w_write_shard %s %d: %s", nm[a], k, z3w_error());
    /* an older object where the pool's all-fill shard 3 goes: it must be removed, as z3w_write_shard does */
    for (int a = 0; a < 3; a++) { snprintf(d, sizeof d, "%s/b%s/c/3/0", tmp, nm[a]); char c[600]; snprintf(c, sizeof c, "mkdir -p %s && echo stale > %s/0", d, d); CHECK(!system(c), "stale object"); }
    dstat st = {0};
    z3w_pool *P = z3w_pool_create(5);
    CHECK(P && z3w_pool_threads(P) == 5, "pool threads");
    for (int k = 0; k < NS; k++) for (int a = 0; a < 3; a++) CHECK(!z3w_pool_submit(P, B[a], k, 0, 0, data[a][k], done, &st), "submit %s %d", nm[a], k);
    z3w_pool_destroy(P);   /* (finishes the queue) */
    CHECK(atomic_load(&st.calls) == 3 * NS && !atomic_load(&st.bad), "done calls %d bad %d (%s)", atomic_load(&st.calls), atomic_load(&st.bad), st.err);
    for (int a = 0; a < 3; a++) for (int k = 0; k < NS; k++) {
        char pa[600], pb[600]; snprintf(pa, sizeof pa, "%s/a%s/c/%d/0/0", tmp, nm[a], k); snprintf(pb, sizeof pb, "%s/b%s/c/%d/0/0", tmp, nm[a], k);
        long na, nb; uint8_t *x = slurp(pa, &na), *y = slurp(pb, &nb);
        if (k == 3) CHECK(na == -1 && nb == -1, "%s shard 3 (all fill): objects %ld %ld", nm[a], na, nb);
        else CHECK(na > 0 && na == nb && !memcmp(x, y, (size_t)na), "%s shard %d: pool %ld bytes vs z3w_write_shard %ld, or different bytes", nm[a], k, nb, na);
        free(x); free(y);
    }
    /* a failing shard: its z directory is a file; the others written, rc and error text through done */
    snprintf(d, sizeof d, "%s/fq0", tmp); z3w *F = z3w_create(d, shape, S, 0, 0, nullptr);
    { char c[600]; snprintf(c, sizeof c, "mkdir -p %s/c && touch %s/c/1", d, d); CHECK(!system(c), "blocker"); }
    dstat sf = {0}; P = z3w_pool_create(3);
    for (int k = 0; k < 3; k++) CHECK(!z3w_pool_submit(P, F, k, 0, 0, data[0][k], done, &sf), "submit fail-case %d", k);
    z3w_pool_destroy(P);
    CHECK(atomic_load(&sf.calls) == 3 && atomic_load(&sf.bad) == 1 && strstr(sf.err, "mkdir"), "failing shard: calls %d bad %d err '%s'", atomic_load(&sf.calls), atomic_load(&sf.bad), sf.err);
    { char p0[600], p2[600]; struct stat sb; snprintf(p0, sizeof p0, "%s/c/0/0/0", d); snprintf(p2, sizeof p2, "%s/c/2/0/0", d); CHECK(!stat(p0, &sb) && !stat(p2, &sb), "the other shards beside the failing one"); }
    for (int a = 0; a < 3; a++) { z3w_close(A[a]); z3w_close(B[a]); for (int k = 0; k < NS; k++) free(data[a][k]); }
    z3w_close(F);
    { char c[600]; snprintf(c, sizeof c, "rm -rf -- '%s'", tmp); if (strstr(tmp, "/test_z3w_pool.")) (void)!system(c); }
    if (fails) { fprintf(stderr, "test_z3w_pool: %d failures\n", fails); return 1; }
    printf("test_z3w_pool: ok (3 arrays x %d shards, pool == z3w_write_shard byte for byte; failing shard reported)\n", NS);
    return 0;
}
