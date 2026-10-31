/* ufsm fields-pack IN OUT [--q-phase Q] [--q Q] [--ang A] [--conf C] [--mask-w 0|1] [--shard 512] [--threads N]
   Re-encodes a winding fit-field store (predict --fit-zarr: pc ps t0..t5 sp w [r dz dy dx dq ct cs] + fields.json) smaller, for a
   pod to run before copying its fields home. The output is an ordinary fit-field store (uint8 volcomp zarr v3 arrays of the same
   names and meaning), so every reader takes it unchanged.
     --q-phase Q   volcomp quantiser of the phase planes pc / ps (default 4, predict's default since 10-30; 0 lossless)
     --q Q         volcomp quantiser of the other planes (default -1: their chunks copied byte for byte, only re-sharded)
     --ang A       the phase angle rounded to A steps per turn before coding (0: off)
     --conf C      the phase confidence |pc, ps| rounded to C steps (0: off)
     --mask-w 1    pc = ps = 128 (no information) where w = 0 (every reader ignores the phase there; default 0: no measurable gain, the predictor already writes 128 where nothing was pooled)
   Planes copied byte for byte keep their chunks; re-encoded ones are decoded shard by shard (one worker per shard). Shards and
   chunks equal to the fill value are omitted as z3w does. Other files of IN (fields.json, ...) are copied. */
#include "json.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

int repack_array(const char *in, const char *out, int shard, int nth, int quiet);   /* src/repack.c */

typedef struct {
    const char *in, *out; int shard; float qph, q; int ang, conf, maskw;
    z3w *zw; const char *plane; int phase;   /* the job: one plane (phase: pc and ps together, zw2 the ps writer) */
    z3w *zw2; int64_t shape[3], ns[3]; atomic_long next; atomic_int fail; uint8_t lut[2][65536]; uint8_t *present;
    int crop; int64_t bo[3], bn[3];   /* a crop (fields.json zarr_offset): only the shards meeting the box [bo, bo + bn) are written */
} fp_job;

static char *fp_slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1); if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return nullptr; }
    fclose(f); b[n] = 0; return b;
}
static int fp_copy(const char *a, const char *b) {
    FILE *f = fopen(a, "rb"), *g = f ? fopen(b, "wb") : nullptr; if (!f || !g) { if (f) fclose(f); return -1; }
    char buf[1 << 16]; size_t n; int bad = 0;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) bad |= fwrite(buf, 1, n, g) != n;
    fclose(f); bad |= fclose(g) != 0; return bad ? -1 : 0;
}
/* the attributes object of an array's zarr.json with a "fields_pack" record put first ("{}" when absent) */
static char *fp_attrs(const char *zj, const fp_job *J, float q) {
    char rec[256]; snprintf(rec, sizeof rec, "\"fields_pack\":{\"q\":%g,\"ang\":%d,\"conf\":%d,\"mask_w\":%d}", (double)q, J->ang, J->conf, J->maskw);
    const char *a = strstr(zj, "\"attributes\""); const char *s = a ? strchr(a, '{') : nullptr; const char *e = s;
    int depth = 0, str = 0;
    for (; e && *e; e++) { if (str) { if (*e == '\\') e++; else if (*e == '"') str = 0; continue; } if (*e == '"') str = 1; else if (*e == '{') depth++; else if (*e == '}' && !--depth) break; }
    const size_t n = s && e && *e ? (size_t)(e - s) : 0; char *o = malloc(n + sizeof rec + 8);
    if (!n) { sprintf(o, "{%s}", rec); return o; }
    const int empty = strspn(s + 1, " \t\r\n") == n - 1;
    sprintf(o, "{%s%s%.*s}", rec, empty ? "" : ",", (int)(n - 1), s + 1);
    return o;
}
static void fp_lut(fp_job *J) {   /* (pc, ps) -> the quantised phase vector */
    for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) {
        double c = (a - 128) / 127.0, s = (b - 128) / 127.0, r = hypot(c, s), t = atan2(s, c);
        if (J->ang > 0) t = round(t / (2 * M_PI) * J->ang) * 2 * M_PI / J->ang;
        if (J->conf > 0) r = round(fmin(r, 1.0) * J->conf) / J->conf;
        int pa = J->ang > 0 || J->conf > 0 ? (int)lround(128 + 127 * r * cos(t)) : a, pb = J->ang > 0 || J->conf > 0 ? (int)lround(128 + 127 * r * sin(t)) : b;
        J->lut[0][a << 8 | b] = (uint8_t)(pa < 0 ? 0 : pa > 255 ? 255 : pa); J->lut[1][a << 8 | b] = (uint8_t)(pb < 0 ? 0 : pb > 255 ? 255 : pb);
    }
}
static void *fp_worker(void *arg) {
    fp_job *J = arg; const int S = J->shard; const size_t n = (size_t)S * S * S;
    store *st = store_open(J->in);
    z3 *za = st ? z3_open(st, J->phase ? "pc" : J->plane, nullptr) : nullptr, *zb = J->phase && st ? z3_open(st, "ps", nullptr) : nullptr,
       *zw = J->phase && J->maskw && st ? z3_open(st, "w", nullptr) : nullptr;
    uint8_t *a = malloc(n), *b = J->phase ? malloc(n) : nullptr, *w = zw ? malloc(n) : nullptr;
    if (!za || (J->phase && !zb) || (J->phase && J->maskw && !zw) || !a || (J->phase && !b) || (zw && !w)) { atomic_store(&J->fail, 1); goto done; }
    for (;;) {
        const long k = atomic_fetch_add(&J->next, 1); if (k >= J->ns[0] * J->ns[1] * J->ns[2] || atomic_load(&J->fail)) break;
        if (J->present && !J->present[k]) continue;
        const int64_t o[3] = {k / (J->ns[1] * J->ns[2]) * S, k / J->ns[2] % J->ns[1] * S, k % J->ns[2] * S}, nn[3] = {S, S, S};
        if (z3_read(za, o, nn, a, 1) || (zb && z3_read(zb, o, nn, b, 1)) || (zw && z3_read(zw, o, nn, w, 1))) { fprintf(stderr, "fields-pack: %s\n", z3_error()); atomic_store(&J->fail, 1); break; }
        if (J->phase) {
            const int q = J->ang > 0 || J->conf > 0;
            for (size_t i = 0; i < n; i++) {
                if (w && !w[i]) { a[i] = b[i] = 128; continue; }
                if (q) { const int x = a[i] << 8 | b[i]; a[i] = J->lut[0][x]; b[i] = J->lut[1][x]; }
            }
        }
        if (z3w_write_shard(J->zw, o[0] / S, o[1] / S, o[2] / S, a, 1) || (J->phase && z3w_write_shard(J->zw2, o[0] / S, o[1] / S, o[2] / S, b, 1))) {
            fprintf(stderr, "fields-pack: %s\n", z3w_error()); atomic_store(&J->fail, 1); break; }
    }
done:
    free(a); free(b); free(w); if (za) z3_close(za); if (zb) z3_close(zb); if (zw) z3_close(zw); if (st) store_close(st);
    return nullptr;
}
static long long fp_du(const char *dir) {   /* bytes under an array directory (files only) */
    DIR *d = opendir(dir); if (!d) return 0; long long s = 0; struct dirent *e; char p[2048];
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name); struct stat sb;
        if (lstat(p, &sb)) continue;
        s += S_ISDIR(sb.st_mode) ? fp_du(p) : (long long)sb.st_size;
    }
    closedir(d); return s;
}
/* the shards of the output grid holding any stored chunk of the array (1) */
static uint8_t *fp_present(const char *in, const char *plane, const int64_t ns[3], int S) {
    store *st = store_open(in); z3 *z = st ? z3_open(st, plane, nullptr) : nullptr; int64_t g[3]; uint8_t *bits = nullptr;
    uint8_t *p = nullptr;
    if (z && z3_chunk_map(z, g, &bits) >= 0) {
        p = calloc((size_t)(ns[0] * ns[1] * ns[2]), 1); const int k = S / 128;
        for (int64_t a = 0; a < g[0]; a++) for (int64_t b = 0; b < g[1]; b++) for (int64_t c = 0; c < g[2]; c++)
            if (bits[(a * g[1] + b) * g[2] + c]) p[((a / k) * ns[1] + b / k) * ns[2] + c / k] = 1;
    }
    free(bits); if (z) z3_close(z); if (st) store_close(st); return p;
}

int cmd_fields_pack(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fields-pack IN OUT [--q-phase 4] [--q -1] [--ang 0] [--conf 0] [--mask-w 1] [--shard 512] [--threads N]\n"); return 2; }
    static fp_job J; memset(&J, 0, sizeof J);
    J.in = argv[2]; J.out = argv[3]; J.shard = 512; J.qph = 4; J.q = -1; J.maskw = 0; int nth = ufsm_threads(0, 64);
    for (int a = 4; a + 1 < argc; a += 2) {
        const char *k = argv[a], *v = argv[a + 1];
        if (!strcmp(k, "--q-phase")) J.qph = (float)atof(v); else if (!strcmp(k, "--q")) J.q = (float)atof(v);
        else if (!strcmp(k, "--ang")) J.ang = atoi(v); else if (!strcmp(k, "--conf")) J.conf = atoi(v);
        else if (!strcmp(k, "--mask-w")) J.maskw = atoi(v); else if (!strcmp(k, "--shard")) J.shard = atoi(v);
        else if (!strcmp(k, "--threads")) nth = atoi(v);
        else { fprintf(stderr, "fields-pack: unknown option %s\n", k); return 2; }
    }
    if (!(J.qph >= 0 && J.qph <= 255) || J.q > 255 || J.shard <= 0 || J.shard % 128 || nth < 1) { fprintf(stderr, "fields-pack: bad option value\n"); return 2; }
    char p[2048], p2[2048];
    snprintf(p, sizeof p, "%s/fields.json", J.in); { FILE *t = fopen(p, "rb"); if (!t) { fprintf(stderr, "fields-pack: no %s\n", p); return 1; } fclose(t); }
    if (mkdir(J.out, 0755) && errno != EEXIST) { fprintf(stderr, "fields-pack: mkdir %s: %s\n", J.out, strerror(errno)); return 1; }
    fp_lut(&J);
    {   /* a crop (scripts/wbench/mkcrop.sh): zarr_offset + shape */
        char *t = fp_slurp(p); json *j = t ? json_parse(t, strlen(t)) : nullptr; free(t);
        const json *zo = j ? json_get(j, "zarr_offset") : nullptr, *sh = j ? json_get(j, "shape") : nullptr;
        if (zo && sh) { J.crop = 1; for (int d = 0; d < 3; d++) { J.bo[d] = (int64_t)json_num(json_at(zo, d), 0); J.bn[d] = (int64_t)json_num(json_at(sh, d), 0); } }
        json_free(j);
        if (J.crop) fprintf(stderr, "fields-pack: a crop: only the shards meeting the box %lld,%lld,%lld + %lldx%lldx%lld; unchanged arrays linked\n",
                            (long long)J.bo[0], (long long)J.bo[1], (long long)J.bo[2], (long long)J.bn[0], (long long)J.bn[1], (long long)J.bn[2]);
    }
    DIR *d = opendir(J.in); if (!d) { fprintf(stderr, "fields-pack: %s\n", J.in); return 1; }
    char *names[64]; int nn = 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "%s/%s", J.in, e->d_name); struct stat sb;
        if (stat(p, &sb)) continue;
        if (!S_ISDIR(sb.st_mode)) { snprintf(p2, sizeof p2, "%s/%s", J.out, e->d_name); if (fp_copy(p, p2)) { fprintf(stderr, "fields-pack: copy %s\n", p); closedir(d); return 1; } continue; }
        snprintf(p, sizeof p, "%s/%s/zarr.json", J.in, e->d_name); struct stat zs;
        if (stat(p, &zs)) { fprintf(stderr, "fields-pack: %s/%s is not an array: skipped\n", J.in, e->d_name); continue; }
        if (nn < 64) names[nn++] = strdup(e->d_name);
    }
    closedir(d);
    int has_pc = 0, has_ps = 0, has_w = 0; for (int i = 0; i < nn; i++) { has_pc |= !strcmp(names[i], "pc"); has_ps |= !strcmp(names[i], "ps"); has_w |= !strcmp(names[i], "w"); }
    if (J.maskw && !has_w) J.maskw = 0;
    long long tin = 0, tout = 0; int rc = 0;
    for (int i = 0; i < nn && !rc; i++) {
        const char *nm = names[i]; const int ph = !strcmp(nm, "pc") || !strcmp(nm, "ps");
        if (!strcmp(nm, "ps") && has_pc) continue;   /* (with pc) */
        const int phase = ph && has_pc && has_ps;
        const float q = ph ? J.qph : J.q;
        /* an array already at the target quantiser (e.g. a store predict wrote at the new defaults) is copied, not re-encoded */
        float qin = -1; { store *st = store_open(J.in); z3 *z = st ? z3_open(st, nm, nullptr) : nullptr; if (z) { qin = z3_meta_of(z)->q; z3_close(z); } if (st) store_close(st); }
        const int recode = ph ? phase && ((J.qph > 0 && qin != J.qph) || J.ang > 0 || J.conf > 0 || J.maskw) : J.q >= 0 && qin != J.q;
        snprintf(p, sizeof p, "%s/%s", J.in, nm); snprintf(p2, sizeof p2, "%s/%s", J.out, nm);
        const long long bi = fp_du(p) + (phase ? (snprintf(p2, sizeof p2, "%s/ps", J.in), fp_du(p2)) : 0);
        snprintf(p2, sizeof p2, "%s/%s", J.out, nm);
        if (!recode && J.crop) {   /* a crop: the unchanged array linked */
            char rp[PATH_MAX]; rc = !realpath(p, rp) || symlink(rp, p2);
            if (phase && !rc) { snprintf(p, sizeof p, "%s/ps", J.in); snprintf(p2, sizeof p2, "%s/ps", J.out); rc = !realpath(p, rp) || symlink(rp, p2); }
            if (rc) fprintf(stderr, "fields-pack: cannot link %s\n", p);
        } else if (!recode) {   /* chunks byte for byte, re-sharded */
            rc = repack_array(p, p2, J.shard, nth, 1) != 0;
            if (phase && !rc) { snprintf(p, sizeof p, "%s/ps", J.in); snprintf(p2, sizeof p2, "%s/ps", J.out); rc = repack_array(p, p2, J.shard, nth, 1) != 0; }
        } else {
            snprintf(p, sizeof p, "%s/%s/zarr.json", J.in, nm); char *zj = fp_slurp(p); if (!zj) { rc = 1; break; }
            store *st = store_open(J.in); z3 *z = st ? z3_open(st, nm, nullptr) : nullptr;
            if (!z) { fprintf(stderr, "fields-pack: cannot open %s/%s (uint8 zarr arrays only)\n", J.in, nm); free(zj); rc = 1; break; }
            const z3_meta *m = z3_meta_of(z); memcpy(J.shape, m->shape, sizeof J.shape); const int fill = m->fill; z3_close(z); store_close(st);
            for (int k = 0; k < 3; k++) J.ns[k] = (J.shape[k] + J.shard - 1) / J.shard;
            char *at = fp_attrs(zj, &J, q); free(zj);
            J.zw = z3w_create(p2, J.shape, J.shard, q, fill, at); J.zw2 = nullptr;
            if (phase) {
                snprintf(p, sizeof p, "%s/ps/zarr.json", J.in); char *zj2 = fp_slurp(p); char *at2 = zj2 ? fp_attrs(zj2, &J, q) : nullptr; free(zj2);
                snprintf(p, sizeof p, "%s/ps", J.out); J.zw2 = at2 ? z3w_create(p, J.shape, J.shard, q, 128, at2) : nullptr; free(at2);
            }
            free(at);
            if (!J.zw || (phase && !J.zw2)) { fprintf(stderr, "fields-pack: %s\n", z3w_error()); rc = 1; break; }
            J.plane = nm; J.phase = phase; atomic_store(&J.next, 0); atomic_store(&J.fail, 0);
            J.present = fp_present(J.in, nm, J.ns, J.shard);
            if (phase && J.present) { uint8_t *p2b = fp_present(J.in, "ps", J.ns, J.shard); if (p2b) for (int64_t k = 0; k < J.ns[0] * J.ns[1] * J.ns[2]; k++) J.present[k] |= p2b[k]; else { free(J.present); J.present = nullptr; } free(p2b); }
            if (J.crop) {
                if (!J.present) J.present = memset(malloc((size_t)(J.ns[0] * J.ns[1] * J.ns[2])), 1, (size_t)(J.ns[0] * J.ns[1] * J.ns[2]));
                for (int64_t a = 0; a < J.ns[0]; a++) for (int64_t b = 0; b < J.ns[1]; b++) for (int64_t c = 0; c < J.ns[2]; c++) {
                    const int64_t s3[3] = {a, b, c}; int in = 1;
                    for (int d = 0; d < 3; d++) in &= s3[d] * J.shard < J.bo[d] + J.bn[d] && (s3[d] + 1) * J.shard > J.bo[d];
                    if (!in) J.present[(a * J.ns[1] + b) * J.ns[2] + c] = 0;
                }
            }
            const int nw = nth < 64 ? nth : 64; pthread_t th[64];
            for (int t = 0; t < nw; t++) pthread_create(&th[t], nullptr, fp_worker, &J);
            for (int t = 0; t < nw; t++) pthread_join(th[t], nullptr);
            free(J.present); J.present = nullptr;
            rc = atomic_load(&J.fail) || z3w_close(J.zw) || (J.zw2 && z3w_close(J.zw2));
        }
        snprintf(p2, sizeof p2, "%s/%s", J.out, nm); long long bo = fp_du(p2); if (phase) { snprintf(p2, sizeof p2, "%s/ps", J.out); bo += fp_du(p2); }
        fprintf(stderr, "fields-pack: %-5s %s %8.1f MB -> %8.1f MB (%.2fx)%s\n", phase ? "pc+ps" : nm, recode ? "q" : "copy", bi / 1e6, bo / 1e6, bo ? (double)bi / bo : 0.0, rc ? "  FAILED" : "");
        tin += bi; tout += bo;
    }
    for (int i = 0; i < nn; i++) free(names[i]);
    if (rc) { fprintf(stderr, "fields-pack: FAILED\n"); return 1; }
    printf("fields-pack %s -> %s: %.2f GB -> %.2f GB (%.2fx); phase q %g ang %d conf %d mask-w %d, other planes %s\n", J.in, J.out, tin / 1e9, tout / 1e9,
           tout ? (double)tin / tout : 0.0, (double)J.qph, J.ang, J.conf, J.maskw, J.q >= 0 ? "re-encoded" : "copied");
    return 0;
}
