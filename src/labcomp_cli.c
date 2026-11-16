/* ufsm labcomp-pack IN_ARRAY_DIR OUT_ARRAY_DIR [--kind codes] [--threads N] [--verify 1]
   A uint8 label array (any codec z3_read takes) re-encoded with the labcomp codec (labcomp.h) into a new array of the same shape,
   shards, fill and attributes (plus a "labcomp" record). Only stored chunks are visited (the chunk map); every encoded chunk is decoded
   again and compared byte for byte (--verify 0: off), so a written store is exact. The input is never modified; OUT must not exist.
   A group: run once per level and copy the group zarr.json (the readers find levels by the group's multiscales). */
#include "labcomp.h"
#include "optcheck.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}

typedef struct {
    z3 *in; z3w *out; int kind, verify, cg; int64_t sgrid[3], cgrid[3]; uint8_t *bits; atomic_long next; atomic_int fail;
    atomic_ullong in_bytes, out_bytes, chunks; int64_t nshards;
} lp_job;

static void *lp_worker(void *arg) {
    lp_job *J = arg; const int cg = J->cg, nc = cg * cg * cg;
    uint8_t *raw = malloc(LBC_VOXELS), *chk = malloc(LBC_VOXELS); uint8_t **enc = calloc((size_t)nc, sizeof *enc); size_t *len = calloc((size_t)nc, sizeof *len);
    if (!raw || !chk || !enc || !len) { atomic_store(&J->fail, 1); goto out; }
    for (;;) {
        const long s = atomic_fetch_add(&J->next, 1); if (s >= J->nshards || atomic_load(&J->fail)) break;
        const int64_t sz = s / (J->sgrid[1] * J->sgrid[2]), sy = s / J->sgrid[2] % J->sgrid[1], sx = s % J->sgrid[2];
        int any = 0;
        for (int i = 0; i < nc && !atomic_load(&J->fail); i++) {
            const int64_t cz = sz * cg + i / (cg * cg), cy = sy * cg + i / cg % cg, cx = sx * cg + i % cg;
            if (cz >= J->cgrid[0] || cy >= J->cgrid[1] || cx >= J->cgrid[2] || !J->bits[(cz * J->cgrid[1] + cy) * J->cgrid[2] + cx]) continue;
            uint8_t *rb = nullptr; size_t rn = 0; if (z3_chunk_raw(J->in, cz, cy, cx, &rb, &rn) > 0) atomic_fetch_add(&J->in_bytes, rn); free(rb);
            const int64_t o[3] = {cz * 128, cy * 128, cx * 128}, n[3] = {128, 128, 128};
            if (z3_read(J->in, o, n, raw, 1)) { fprintf(stderr, "labcomp-pack: read: %s\n", z3_error()); atomic_store(&J->fail, 1); break; }
            const int fill = z3_meta_of(J->in)->fill; int nonfill = 0;
            for (size_t k = 0; k < LBC_VOXELS && !nonfill; k++) nonfill = raw[k] != fill;
            if (!nonfill) continue;
            if (lbc_codes_encode(raw, &enc[i], &len[i])) { fprintf(stderr, "labcomp-pack: encode failed\n"); atomic_store(&J->fail, 1); break; }
            if (J->verify && (lbc_decode(enc[i], len[i], chk) || memcmp(chk, raw, LBC_VOXELS))) {
                fprintf(stderr, "labcomp-pack: chunk %lld %lld %lld does not decode exactly\n", (long long)cz, (long long)cy, (long long)cx); atomic_store(&J->fail, 1); break; }
            atomic_fetch_add(&J->out_bytes, len[i]); atomic_fetch_add(&J->chunks, 1); any = 1;
        }
        if (atomic_load(&J->fail)) break;
        if (any && z3w_write_shard_enc(J->out, sz, sy, sx, enc, len)) { fprintf(stderr, "labcomp-pack: write: %s\n", z3w_error()); atomic_store(&J->fail, 1); break; }
        for (int i = 0; i < nc; i++) { free(enc[i]); enc[i] = nullptr; len[i] = 0; }
    }
out:
    if (enc) for (int i = 0; i < nc; i++) free(enc[i]);
    free(raw); free(chk); free(enc); free(len); return nullptr;
}

static char *lp_attrs(const char *zj, const char *rec) {   /* the attributes object of zarr.json text with rec put first */
    const char *a = strstr(zj, "\"attributes\""); const char *s = a ? strchr(a, '{') : nullptr; const char *e = s; int depth = 0, str = 0;
    for (; e && *e; e++) { if (str) { if (*e == '\\') e++; else if (*e == '"') str = 0; continue; } if (*e == '"') str = 1; else if (*e == '{') depth++; else if (*e == '}' && !--depth) break; }
    const size_t n = s && e && *e ? (size_t)(e - s) : 0; char *o = malloc(n + strlen(rec) + 8); if (!o) return nullptr;
    if (!n) { sprintf(o, "{%s}", rec); return o; }
    const int empty = strspn(s + 1, " \t\r\n") == n - 1;
    sprintf(o, "{%s%s%.*s}", rec, empty ? "" : ",", (int)(n - 1), s + 1); return o;
}

int cmd_labcomp_pack(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm labcomp-pack IN_ARRAY_DIR OUT_ARRAY_DIR [--kind codes] [--threads N] [--verify 1]\n"); return 2; }
    const char *in = argv[2], *outd = argv[3], *kind = opt(argc, argv, "--kind", "codes");
    if (strcmp(kind, "codes")) { fprintf(stderr, "labcomp-pack: --kind codes (the only kind so far)\n"); return 2; }
    struct stat st; if (!stat(outd, &st)) { fprintf(stderr, "labcomp-pack: %s exists (never overwritten)\n", outd); return 1; }
    char root[1200], key[400]; const char *sl = strrchr(in, '/'); if (!sl) { fprintf(stderr, "labcomp-pack: IN must be a path\n"); return 2; }
    snprintf(root, sizeof root, "%.*s", (int)(sl - in), in); snprintf(key, sizeof key, "%s", sl + 1);
    store *s = store_open(root); z3 *z = s ? z3_open(s, key, nullptr) : nullptr;
    if (!z) { fprintf(stderr, "labcomp-pack: %s: %s\n", in, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    if (m->chunk[0] != 128 || m->chunk[1] != 128 || m->chunk[2] != 128 || m->esize != 1 || !m->sharded || m->shard[0] % 128) { fprintf(stderr, "labcomp-pack: uint8, sharded, 128^3 chunks only\n"); return 1; }
    lp_job J; memset(&J, 0, sizeof J); J.in = z; J.kind = LBC_KIND_CODES; J.verify = opt_atoi("--verify", opt(argc, argv, "--verify", "1")); J.cg = m->shard[0] / 128;
    if (z3_chunk_map(z, J.cgrid, &J.bits) < 0) { fprintf(stderr, "labcomp-pack: chunk map: %s\n", z3_error()); return 1; }
    for (int d = 0; d < 3; d++) J.sgrid[d] = (m->shape[d] + m->shard[0] - 1) / m->shard[0];
    J.nshards = J.sgrid[0] * J.sgrid[1] * J.sgrid[2];
    char zp[1400]; snprintf(zp, sizeof zp, "%s/zarr.json", in); FILE *f = fopen(zp, "rb"); char zj[1 << 16] = {0}; if (f) { (void)!fread(zj, 1, sizeof zj - 1, f); fclose(f); }
    char rec[200]; snprintf(rec, sizeof rec, "\"labcomp\":{\"kind\":\"%s\",\"from\":\"volcomp-or-other\",\"exact\":true}", kind);
    char *attrs = lp_attrs(zj, rec);
    J.out = z3w_create_codec(outd, m->shape, m->shard[0], lbc_codec_json(LBC_KIND_CODES), m->fill, attrs); free(attrs);
    if (!J.out) { fprintf(stderr, "labcomp-pack: %s: %s\n", outd, z3w_error()); return 1; }
    const int nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 8);
    pthread_t th[256]; const int n = nt < 256 ? nt : 256;
    for (int i = 0; i < n; i++) pthread_create(&th[i], nullptr, lp_worker, &J);
    for (int i = 0; i < n; i++) pthread_join(th[i], nullptr);
    z3w_close(J.out); free(J.bits);
    const unsigned long long ib = atomic_load(&J.in_bytes), ob = atomic_load(&J.out_bytes), nc = atomic_load(&J.chunks);
    printf("labcomp-pack %s: %llu chunks, in %.3f GB -> out %.3f GB (%.2fx smaller, %.1fx vs raw)%s\n", in, nc, ib / 1e9, ob / 1e9, ob ? (double)ib / ob : 0,
           ob ? (double)nc * LBC_VOXELS / ob : 0, atomic_load(&J.fail) ? " FAILED" : J.verify ? ", every chunk verified exact" : "");
    z3_close(z); store_close(s);
    return atomic_load(&J.fail) ? 1 : 0;
}
