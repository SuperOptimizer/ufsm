/* ufsm repack <in-array-dir> <out-array-dir> <shard> [--threads N]
   Lossless re-sharding of a z3w-written zarr v3 uint8 array (sharding_indexed, 128^3 inner chunks, index at the end with crc32c):
   the compressed inner chunks are copied byte for byte into shards of <shard>^3 (a multiple of 128) with a new index; nothing is
   decoded or re-encoded. zarr.json is copied with only the outer chunk_shape changed. Shards whose chunks are all absent (fill) are
   omitted, as z3w does. E.g. predict --fit-zarr writes 128^3 shards (one chunk per file); 512^3 shards hold up to 64 chunks per file,
   ~64x fewer files to upload. */
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

uint32_t crc32c(uint32_t crc, const void *buf, size_t len);   /* src/z3w.c */

static int mkdirs(const char *path) {
    char p[1400]; snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(p, 0755) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(p, 0755) && errno != EEXIST ? -1 : 0;
}
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)len + 1);
    if (!b || fread(b, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(b); return nullptr; }
    fclose(f); b[len] = 0; if (n) *n = (size_t)len; return b;
}

typedef struct {
    const char *in, *out; int64_t shape[3]; int sin, sout, kin, kout;   /* shard sizes, inner chunks per shard axis */
    int64_t nout[3]; atomic_long next; atomic_int failed; atomic_long files, chunks, bytes;
} rp_t;

/* the encoded chunk (cz, cy, cx) (global inner-chunk coordinates) from its input shard: malloc'd bytes, *len; nullptr + *len 0 if absent */
static uint8_t *read_chunk(rp_t *R, int64_t cz, int64_t cy, int64_t cx, size_t *len, int *err) {
    *len = 0;
    char p[1500]; snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", R->in, (long long)(cz / R->kin), (long long)(cy / R->kin), (long long)(cx / R->kin));
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;   /* absent shard: fill */
    const int nc = R->kin * R->kin * R->kin; const size_t isz = (size_t)nc * 16 + 4;
    uint8_t *idx = malloc(isz); uint8_t *buf = nullptr;
    if (fseek(f, -(long)isz, SEEK_END) || fread(idx, 1, isz, f) != isz) { *err = 1; goto done; }
    uint32_t crc; memcpy(&crc, idx + (size_t)nc * 16, 4);
    if (crc32c(0, idx, (size_t)nc * 16) != crc) { fprintf(stderr, "repack: index crc mismatch in %s\n", p); *err = 1; goto done; }
    const int i = (int)(((cz % R->kin) * R->kin + cy % R->kin) * R->kin + cx % R->kin);
    uint64_t off, n; memcpy(&off, idx + (size_t)i * 16, 8); memcpy(&n, idx + (size_t)i * 16 + 8, 8);
    if (off == UINT64_MAX && n == UINT64_MAX) goto done;
    buf = malloc(n ? n : 1);
    if (fseek(f, (long)off, SEEK_SET) || fread(buf, 1, n, f) != n) { free(buf); buf = nullptr; *err = 1; goto done; }
    *len = n;
done:
    free(idx); fclose(f); return buf;
}

static void *rp_worker(void *arg) {
    rp_t *R = arg;
    const int nc = R->kout * R->kout * R->kout;
    uint8_t **enc = calloc((size_t)nc, sizeof *enc); size_t *len = calloc((size_t)nc, sizeof *len);
    uint8_t *idx = malloc((size_t)nc * 16 + 4);
    const int64_t total = R->nout[0] * R->nout[1] * R->nout[2];
    for (;;) {
        const int64_t s = atomic_fetch_add(&R->next, 1); if (s >= total || atomic_load(&R->failed)) break;
        const int64_t oz = s / (R->nout[1] * R->nout[2]), oy = (s / R->nout[2]) % R->nout[1], ox = s % R->nout[2];
        int any = 0, err = 0;
        for (int i = 0; i < nc; i++) {
            const int64_t cz = oz * R->kout + i / (R->kout * R->kout), cy = oy * R->kout + (i / R->kout) % R->kout, cx = ox * R->kout + i % R->kout;
            enc[i] = nullptr; len[i] = 0;
            if (cz * 128 >= R->shape[0] || cy * 128 >= R->shape[1] || cx * 128 >= R->shape[2]) continue;   /* outside the array */
            enc[i] = read_chunk(R, cz, cy, cx, &len[i], &err);
            any |= len[i] > 0;
        }
        if (err) { atomic_store(&R->failed, 1); }
        else if (any) {
            char d[1500], p[1600], tmp[1700];
            snprintf(d, sizeof d, "%s/c/%lld/%lld", R->out, (long long)oz, (long long)oy);
            snprintf(p, sizeof p, "%s/%lld", d, (long long)ox); snprintf(tmp, sizeof tmp, "%s.tmp", p);
            FILE *f = mkdirs(d) ? nullptr : fopen(tmp, "wb");
            if (!f) { fprintf(stderr, "repack: cannot write %s\n", tmp); atomic_store(&R->failed, 1); }
            else {
                uint64_t off = 0; long nchunks = 0;
                for (int i = 0; i < nc; i++) {
                    const uint64_t o = len[i] ? off : UINT64_MAX, l = len[i] ? len[i] : UINT64_MAX;
                    memcpy(idx + (size_t)i * 16, &o, 8); memcpy(idx + (size_t)i * 16 + 8, &l, 8);
                    if (len[i]) { if (fwrite(enc[i], 1, len[i], f) != len[i]) atomic_store(&R->failed, 1); off += len[i]; nchunks++; }
                }
                const uint32_t crc = crc32c(0, idx, (size_t)nc * 16); memcpy(idx + (size_t)nc * 16, &crc, 4);
                if (fwrite(idx, 1, (size_t)nc * 16 + 4, f) != (size_t)nc * 16 + 4) atomic_store(&R->failed, 1);
                if (fclose(f) || rename(tmp, p)) atomic_store(&R->failed, 1);
                atomic_fetch_add(&R->files, 1); atomic_fetch_add(&R->chunks, nchunks); atomic_fetch_add(&R->bytes, (long)(off + (size_t)nc * 16 + 4));
            }
        }
        for (int i = 0; i < nc; i++) free(enc[i]);
    }
    free(enc); free(len); free(idx);
    return nullptr;
}

/* the integers of the JSON array following key (first occurrence after from): returns the position after it, or nullptr */
static const char *json_ints(const char *from, const char *key, int64_t v[3]) {
    const char *p = strstr(from, key); if (!p) return nullptr;
    p = strchr(p, '['); if (!p) return nullptr;
    if (sscanf(p, "[%lld,%lld,%lld]", (long long *)&v[0], (long long *)&v[1], (long long *)&v[2]) != 3) return nullptr;
    return strchr(p, ']');
}

int cmd_repack(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm repack <in-array-dir> <out-array-dir> <shard> [--threads N]\n"); return 2; }
    rp_t R = {.in = argv[2], .out = argv[3], .sout = atoi(argv[4])};
    int nth = 32; for (int a = 5; a + 1 < argc; a++) if (!strcmp(argv[a], "--threads")) nth = atoi(argv[a + 1]);
    char p[1400]; snprintf(p, sizeof p, "%s/zarr.json", R.in);
    size_t jn; char *js = slurp(p, &jn); if (!js) { fprintf(stderr, "repack: cannot read %s\n", p); return 1; }
    int64_t cs[3], inner[3];
    const char *q = json_ints(js, "\"shape\"", R.shape);
    const char *g = q ? strstr(q, "\"chunk_grid\"") : nullptr;
    const char *e = g ? json_ints(g, "\"chunk_shape\"", cs) : nullptr;
    const char *sh = e ? strstr(e, "\"sharding_indexed\"") : nullptr;
    if (!e || !sh || !json_ints(sh, "\"chunk_shape\"", inner) || strstr(js, "\"data_type\":\"uint8\"") == nullptr) {
        fprintf(stderr, "repack: %s is not a z3w uint8 sharded array\n", p); return 1; }
    if (inner[0] != 128 || inner[1] != 128 || inner[2] != 128 || cs[0] != cs[1] || cs[0] != cs[2] || cs[0] % 128 || R.sout <= 0 || R.sout % 128) {
        fprintf(stderr, "repack: needs 128^3 inner chunks, cubic shards and a shard size that is a multiple of 128\n"); return 1; }
    R.sin = (int)cs[0]; R.kin = R.sin / 128; R.kout = R.sout / 128;
    for (int d = 0; d < 3; d++) R.nout[d] = (R.shape[d] + R.sout - 1) / R.sout;
    /* zarr.json: the outer chunk_shape (inside "chunk_grid") replaced */
    const char *lb = strchr(g, '['), *rb = e;   /* g..e: "chunk_grid" ... [a,b,c] */
    if (mkdirs(R.out)) { fprintf(stderr, "repack: mkdir %s: %s\n", R.out, strerror(errno)); return 1; }
    snprintf(p, sizeof p, "%s/zarr.json", R.out);
    FILE *f = fopen(p, "w"); if (!f) { fprintf(stderr, "repack: cannot write %s\n", p); return 1; }
    fwrite(js, 1, (size_t)(lb - js), f); fprintf(f, "[%d,%d,%d", R.sout, R.sout, R.sout); fputs(rb, f); fclose(f);
    free(js);
    nth = nth < 1 ? 1 : nth > 256 ? 256 : nth;
    pthread_t th[256];
    for (int i = 0; i < nth; i++) pthread_create(&th[i], nullptr, rp_worker, &R);
    for (int i = 0; i < nth; i++) pthread_join(th[i], nullptr);
    if (atomic_load(&R.failed)) { fprintf(stderr, "repack: FAILED\n"); return 1; }
    printf("repack %s -> %s: shard %d -> %d, %ld files, %ld chunks, %ld bytes\n", R.in, R.out, R.sin, R.sout,
           atomic_load(&R.files), atomic_load(&R.chunks), atomic_load(&R.bytes));
    return 0;
}
