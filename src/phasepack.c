/* ufsm phase-pack / phase-unpack / phase-cmp: compact transfer of a band-field phase label zarr (src/band.h PHASE_*: 0..251, 255
   unknown; volcomp q0 512^3 shards) between machines over a slow link.
   pack:   the shards of one z row of shards -> 6-bit phase (q = round(v / 4) mod 63, cyclic error <= 2 of 252; 255 -> 63), delta along
           x mod 63 against the previous known voxel of the line (unknown stays 63), zstd (level, threads) per shard -> one file.
   unpack: the file -> the same shards (values q * 4, 255 kept) written with z3w (volcomp q0, tmp + rename per shard) into the array
           directory; an existing zarr.json there is kept byte for byte, so readers see the original metadata.
   cmp:    two arrays over one z row of shards: max cyclic error over the known voxels, unknown mismatches.
   Lossy volcomp (q > 0) is not used: the 251 -> 0 wrap of the phase would smear. Built into ufsm and as build/phase_pack (no CUDA). */
#include "zarr3.h"
#include "sysinfo.h"
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <zstd.h>

#define PP_PERIOD 252
#define PP_Q 63          /* 6-bit codes 0..62 = phase / 4, 63 = unknown */
#define PP_UNK 255

static const char MAGIC[8] = "UFPP1\n\0";
static double pnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *popt(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

/* in-place along each x line of length nx */
void phase_pack_lines(uint8_t *v, size_t nlines, int nx) {
    for (size_t l = 0; l < nlines; l++) {
        uint8_t *a = v + l * (size_t)nx; int p = 0;
        for (int x = 0; x < nx; x++) {
            if (a[x] == PP_UNK) { a[x] = PP_Q; continue; }
            const int q = ((a[x] + 2) >> 2) % PP_Q;
            a[x] = (uint8_t)((q - p + PP_Q) % PP_Q); p = q;
        }
    }
}
void phase_unpack_lines(uint8_t *v, size_t nlines, int nx) {
    for (size_t l = 0; l < nlines; l++) {
        uint8_t *a = v + l * (size_t)nx; int p = 0;
        for (int x = 0; x < nx; x++) {
            if (a[x] >= PP_Q) { a[x] = PP_UNK; continue; }
            p = (p + a[x]) % PP_Q; a[x] = (uint8_t)(p * 4);
        }
    }
}

typedef struct { int64_t s[3]; int32_t n[3]; int32_t pad; uint64_t clen; } pp_rec;
typedef struct { char magic[8]; int64_t shape[3]; int32_t shard, fill; } pp_head;

static z3 *open_arr(const char *root, const char *key, store **sp) {
    store *s = store_open(root);
    if (!s) { fprintf(stderr, "phase-pack: cannot open %s\n", root); return nullptr; }
    z3 *z = z3_open(s, key, nullptr);
    if (!z) { fprintf(stderr, "phase-pack: %s/%s: %s\n", root, key, z3_error()); store_close(s); return nullptr; }
    *sp = s; return z;
}

/* page cache: the files this tool reads (the source shards, a sent file) and writes are dropped from the page cache after use
   (posix_fadvise DONTNEED; written data synced first), so a pass over 100+ GB of labels does not evict the trainers' working set */
static void drop_cache(const char *path) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd);
}
static void drop_file(FILE *f) { fflush(f); fdatasync(fileno(f)); posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED); }

static int cmd_pack(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: phase-pack ROOT KEY OUT --row R [--level 19] [--threads 8]\n"); return 2; }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int row = atoi(popt(argc, argv, "--row", "-1")), lev = atoi(popt(argc, argv, "--level", "19")), th = ufsm_threads(atoi(popt(argc, argv, "--threads", "0")), 8);
    store *s; z3 *z = open_arr(root, key, &s); if (!z) return 1;
    const z3_meta *m = z3_meta_of(z);
    const int S = m->shard[0];
    if (!m->sharded || m->shard[1] != S || m->shard[2] != S || m->esize != 1) { fprintf(stderr, "phase-pack: needs a uint8 array with cubic shards\n"); return 1; }
    const int64_t g[3] = {(m->shape[0] + S - 1) / S, (m->shape[1] + S - 1) / S, (m->shape[2] + S - 1) / S};
    if (row < 0 || row >= g[0]) { fprintf(stderr, "phase-pack: --row 0..%lld\n", (long long)g[0] - 1); return 2; }
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE *f = fopen(tmp, "wb"); if (!f) { fprintf(stderr, "phase-pack: cannot write %s\n", tmp); return 1; }
    pp_head h = {.shard = S, .fill = m->fill}; memcpy(h.magic, MAGIC, 8); memcpy(h.shape, m->shape, sizeof h.shape);
    fwrite(&h, sizeof h, 1, f);
    const size_t S3 = (size_t)S * S * S, cap = ZSTD_compressBound(S3);
    uint8_t *buf = malloc(S3), *cb = malloc(cap);
    ZSTD_CCtx *cc = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cc, ZSTD_c_compressionLevel, lev);
    ZSTD_CCtx_setParameter(cc, ZSTD_c_nbWorkers, th > 1 ? th : 0);
    double t0 = pnow(); uint64_t raw = 0, packed = 0; int ns = 0, rc = 0;
    for (int64_t sy = 0; sy < g[1] && !rc; sy++) for (int64_t sx = 0; sx < g[2] && !rc; sx++) {
        const int pr = z3_shard_present(z, row, sy, sx);
        if (pr < 0) { fprintf(stderr, "phase-pack: shard %d/%lld/%lld: %s\n", row, (long long)sy, (long long)sx, z3_error()); rc = 1; break; }
        if (!pr) continue;
        const int64_t o[3] = {(int64_t)row * S, sy * S, sx * S};
        int64_t n[3]; for (int d = 0; d < 3; d++) { n[d] = m->shape[d] - o[d]; if (n[d] > S) n[d] = S; }
        if (z3_read(z, o, n, buf, th)) { fprintf(stderr, "phase-pack: read: %s\n", z3_error()); rc = 1; break; }
        if (strncmp(root, "http", 4)) { char sp[4400]; snprintf(sp, sizeof sp, "%s/%s/c/%d/%lld/%lld", root, key, row, (long long)sy, (long long)sx); drop_cache(sp); }
        const size_t nv = (size_t)n[0] * n[1] * n[2];
        phase_pack_lines(buf, (size_t)n[0] * n[1], (int)n[2]);
        const size_t cl = ZSTD_compress2(cc, cb, cap, buf, nv);
        if (ZSTD_isError(cl)) { fprintf(stderr, "phase-pack: zstd: %s\n", ZSTD_getErrorName(cl)); rc = 1; break; }
        pp_rec r = {.s = {row, sy, sx}, .n = {(int32_t)n[0], (int32_t)n[1], (int32_t)n[2]}, .clen = cl};
        if (fwrite(&r, sizeof r, 1, f) != 1 || fwrite(cb, 1, cl, f) != cl) { fprintf(stderr, "phase-pack: write failed\n"); rc = 1; break; }
        raw += nv; packed += cl; ns++;
    }
    ZSTD_freeCCtx(cc); free(buf); free(cb);
    drop_file(f);
    if (fclose(f)) rc = 1;
    if (!rc && rename(tmp, out)) { fprintf(stderr, "phase-pack: rename %s\n", out); rc = 1; }
    fprintf(stderr, "phase-pack %s row %d: %d shards, %.1f MB raw -> %.1f MB (%.1f%%) in %.0f s\n", key, row, ns, raw / 1e6, packed / 1e6,
            raw ? 100.0 * packed / raw : 0.0, pnow() - t0);
    z3_close(z); store_close(s);
    return rc;
}

static int cmd_unpack(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: phase-unpack IN DIR [--threads 8]   (DIR: the array directory, e.g. .../phaseA/9.6)\n"); return 2; }
    const char *in = argv[2], *dir = argv[3]; const int th = ufsm_threads(atoi(popt(argc, argv, "--threads", "0")), 8);
    FILE *f = fopen(in, "rb"); if (!f) { fprintf(stderr, "phase-unpack: cannot read %s\n", in); return 1; }
    pp_head h; if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, MAGIC, 8)) { fprintf(stderr, "phase-unpack: %s: not a phase-pack file\n", in); fclose(f); return 1; }
    char zj[4200]; snprintf(zj, sizeof zj, "%s/zarr.json", dir);
    char *keep = nullptr; size_t keepn = 0;
    { FILE *k = fopen(zj, "rb"); if (k) { fseek(k, 0, SEEK_END); long L = ftell(k); fseek(k, 0, SEEK_SET); keep = malloc((size_t)L + 1); keepn = fread(keep, 1, (size_t)L, k); fclose(k); } }
    z3w *w = z3w_create(dir, h.shape, h.shard, 0, h.fill, nullptr);
    if (!w) { fprintf(stderr, "phase-unpack: %s\n", z3w_error()); fclose(f); return 1; }
    if (keep) {   /* restore the original metadata (tmp + rename) */
        char t[4300]; snprintf(t, sizeof t, "%s.pp%d", zj, (int)getpid());
        FILE *k = fopen(t, "wb"); int ok = k && fwrite(keep, 1, keepn, k) == keepn; if (k && fclose(k)) ok = 0;
        if (!ok || rename(t, zj)) { fprintf(stderr, "phase-unpack: cannot restore %s\n", zj); fclose(f); return 1; }
        free(keep);
    }
    const size_t S = (size_t)h.shard, S3 = S * S * S;
    uint8_t *buf = malloc(S3), *full = malloc(S3), *cb = nullptr; size_t cbn = 0;
    ZSTD_DCtx *dc = ZSTD_createDCtx();
    double t0 = pnow(); int ns = 0, rc = 0; pp_rec r;
    while (fread(&r, sizeof r, 1, f) == 1) {
        if (r.clen > cbn) { free(cb); cbn = r.clen; cb = malloc(cbn); }
        if (fread(cb, 1, r.clen, f) != r.clen) { fprintf(stderr, "phase-unpack: truncated\n"); rc = 1; break; }
        const size_t nv = (size_t)r.n[0] * r.n[1] * r.n[2];
        const size_t dl = ZSTD_decompressDCtx(dc, buf, S3, cb, r.clen);
        if (ZSTD_isError(dl) || dl != nv) { fprintf(stderr, "phase-unpack: zstd: %s\n", ZSTD_isError(dl) ? ZSTD_getErrorName(dl) : "size"); rc = 1; break; }
        phase_unpack_lines(buf, (size_t)r.n[0] * r.n[1], r.n[2]);
        memset(full, h.fill, S3);
        for (int z = 0; z < r.n[0]; z++) for (int y = 0; y < r.n[1]; y++)
            memcpy(full + ((size_t)z * S + y) * S, buf + ((size_t)z * r.n[1] + y) * r.n[2], (size_t)r.n[2]);
        if (z3w_write_shard(w, r.s[0], r.s[1], r.s[2], full, th)) { fprintf(stderr, "phase-unpack: %s\n", z3w_error()); rc = 1; break; }
        ns++;
    }
    drop_cache(in);
    fclose(f); ZSTD_freeDCtx(dc); free(buf); free(full); free(cb);
    if (z3w_close(w)) rc = 1;
    fprintf(stderr, "phase-unpack %s -> %s: %d shards in %.0f s%s\n", in, dir, ns, pnow() - t0, rc ? " FAILED" : "");
    return rc;
}

static int cmd_cmp(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: phase-cmp ROOT_A KEY_A ROOT_B KEY_B --row R [--threads 8]\n"); return 2; }
    const int row = atoi(popt(argc, argv, "--row", "-1")), th = ufsm_threads(atoi(popt(argc, argv, "--threads", "0")), 8);
    store *sa, *sb; z3 *a = open_arr(argv[2], argv[3], &sa), *b = a ? open_arr(argv[4], argv[5], &sb) : nullptr; if (!a || !b) return 1;
    const z3_meta *m = z3_meta_of(a); const int S = m->shard[0];
    const int64_t gy = (m->shape[1] + S - 1) / S, gx = (m->shape[2] + S - 1) / S;
    const size_t S3 = (size_t)S * S * S; uint8_t *ba = malloc(S3), *bb = malloc(S3);
    uint64_t known = 0, unk = 0, unk_bad = 0, hist[PP_PERIOD / 2 + 1] = {0}; int maxe = 0, ns = 0, rc = 0;
    for (int64_t sy = 0; sy < gy && !rc; sy++) for (int64_t sx = 0; sx < gx; sx++) {
        if (z3_shard_present(a, row, sy, sx) != 1) continue;
        const int64_t o[3] = {(int64_t)row * S, sy * S, sx * S};
        int64_t n[3]; for (int d = 0; d < 3; d++) { n[d] = m->shape[d] - o[d]; if (n[d] > S) n[d] = S; }
        if (z3_read(a, o, n, ba, th) || z3_read(b, o, n, bb, th)) { fprintf(stderr, "phase-cmp: read: %s\n", z3_error()); rc = 1; break; }
        const size_t nv = (size_t)n[0] * n[1] * n[2];
        for (size_t k = 0; k < nv; k++) {
            if (ba[k] == PP_UNK || bb[k] == PP_UNK) { unk++; unk_bad += ba[k] != bb[k]; continue; }
            int e = abs((int)ba[k] - (int)bb[k]) % PP_PERIOD; if (e > PP_PERIOD / 2) e = PP_PERIOD - e;
            if (e > maxe) maxe = e;
            hist[e]++; known++;
        }
        ns++;
    }
    printf("phase-cmp row %d: %d shards, known %llu, unknown %llu (mismatched %llu), max cyclic error %d, error histogram 0:%llu 1:%llu 2:%llu >2:%llu\n",
           row, ns, (unsigned long long)known, (unsigned long long)unk, (unsigned long long)unk_bad, maxe, (unsigned long long)hist[0],
           (unsigned long long)hist[1], (unsigned long long)hist[2], (unsigned long long)(known - hist[0] - hist[1] - hist[2]));
    free(ba); free(bb);
    return rc ? 1 : (maxe > 2 || unk_bad) ? 3 : 0;
}

int cmd_phase_pack(int argc, char **argv) {
    if (!strcmp(argv[1], "phase-drop")) { for (int i = 2; i < argc; i++) drop_cache(argv[i]); return 0; }   /* phase-drop FILE...: drop files from the page cache */
    if (!strcmp(argv[1], "phase-unpack")) return cmd_unpack(argc, argv);
    if (!strcmp(argv[1], "phase-cmp")) return cmd_cmp(argc, argv);
    return cmd_pack(argc, argv);
}

#ifdef PHASE_PACK_MAIN
int main(int argc, char **argv) {   /* build/phase_pack pack|unpack|cmp ...: the arguments of ufsm phase-pack / phase-unpack / phase-cmp */
    if (argc < 2) { fprintf(stderr, "usage: phase_pack pack|unpack|cmp ...\n"); return 2; }
    char *cmd = !strcmp(argv[1], "unpack") ? "phase-unpack" : !strcmp(argv[1], "drop") ? "phase-drop" : !strcmp(argv[1], "cmp") ? "phase-cmp" : "phase-pack";
    argv[1] = cmd;
    return cmd_phase_pack(argc, argv);
}
#endif
