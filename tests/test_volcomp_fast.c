/* volcomp_fast.h + zshm.h: byte-identity with the reference decoder, and the shared decoded-substream cache.
   1. synthetic lossy chunks (smooth, noisy, flat blocks, edges) at q 1..80: vfx_decode = volcomp_decode, vfx_decode_subs = the
      reference vf_decode_sub for random masks; corrupted / truncated streams: same status, same bytes when both decode.
   2. zshm across processes: 6 forked writers / readers on a tiny cache (constant eviction) with keys whose contents are known:
      every hit has the right bytes; a second process sees the first one's entries.
   3. z3_read / z3_read_sub of a lossy zarr with the cache on (cold, then warm, then with random submasks) = without it. */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#define VOLCOMP_IMPLEMENTATION
#include "volcomp.h"
#include "volcomp_fast.h"
#include "zshm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CV VOLCOMP_CHUNK_VOXELS
static uint64_t lcg(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 17; }
static int fails = 0;
static void check(const char *what, int ok) { printf("volcomp_fast: %-96s %s\n", what, ok ? "ok" : "FAIL"); fails += !ok; }

static void make_chunk(uint8_t *c, int kind, uint64_t seed) {
    uint64_t s = seed;
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
        size_t i = ((size_t)z * 128 + y) * 128 + x; int v;
        switch (kind) {
        case 0: v = 128 + (int)(60 * __builtin_sinf(x * 0.11f + y * 0.07f) * __builtin_cosf(z * 0.05f)) + (int)(lcg(&s) % 9) - 4; break;   /* papyrus-ish */
        case 1: v = (int)(lcg(&s) & 255); break;                                                                                            /* noise */
        case 2: v = ((x >> 4) + (y >> 4) + (z >> 4)) % 3 ? 40 : 40 + (int)(lcg(&s) % 200); break;                                        /* flat + busy blocks */
        default: v = x < 64 ? 0 : (y > 90 ? 255 : 100 + (z & 31)); break;                                                                  /* edges, clipping */
        }
        c[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

static int part1(void) {
    uint8_t *c = malloc(CV), *a = malloc(CV), *b = malloc(CV), *enc = malloc(VOLCOMP_ENCODE_BOUND), *bad = malloc(VOLCOMP_ENCODE_BOUND);
    const float qs[] = {1, 2, 4, 8, 20, 80};
    int full_ok = 1, sub_ok = 1, cor_ok = 1, ncor = 0, ncor_dec = 0; uint64_t s = 99;
    for (int kind = 0; kind < 4; kind++)
        for (int qi = 0; qi < 6; qi++) {
            make_chunk(c, kind, 1000 + kind * 10 + qi);
            size_t n = 0;
            if (volcomp_encode(c, qs[qi], enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) { full_ok = 0; continue; }
            volcomp_status s1 = volcomp_decode(enc, n, a, CV), s2 = vfx_decode(enc, n, b, CV);
            if (s1 != VOLCOMP_OK || s2 != VOLCOMP_OK || memcmp(a, b, CV)) full_ok = 0;
            vf_parsed p;
            if (enc[5] == VLL_MODE_LOSSY && vf_parse(enc, n, &p) == VOLCOMP_OK)
                for (int t = 0; t < 4; t++) {
                    uint32_t mask = (uint32_t)lcg(&s) ^ (uint32_t)(lcg(&s) << 15);
                    memset(a, 0, CV); memset(b, 0, CV);
                    for (uint32_t k = 0; k < VF_NSUB; k++) if (mask >> k & 1) vf_decode_sub(&p, k, a, -1, NULL);
                    if (vfx_decode_subs(&p, mask, b) != VOLCOMP_OK || memcmp(a, b, CV)) sub_ok = 0;
                }
            for (int t = 0; t < 40; t++) {   /* corrupted copies: flipped bytes past the header, or truncated */
                size_t m = n; memcpy(bad, enc, n);
                if (t % 5 == 4) m = 8 + lcg(&s) % (n - 8);
                else for (int f = 0; f < 1 + t % 3; f++) bad[8 + lcg(&s) % (n - 8)] ^= (uint8_t)(1u << (lcg(&s) & 7));
                memset(a, 0x55, CV); memset(b, 0x55, CV);
                s1 = volcomp_decode(bad, m, a, CV); s2 = vfx_decode(bad, m, b, CV);
                ncor++;
                if (s1 != s2 || (s1 == VOLCOMP_OK && memcmp(a, b, CV))) cor_ok = 0;
                if (s1 == VOLCOMP_OK) ncor_dec++;
            }
        }
    char msg[160];
    check("full chunks: vfx_decode = volcomp_decode (4 kinds x q 1, 2, 4, 8, 20, 80)", full_ok);
    check("random substream masks: vfx_decode_subs = vf_decode_sub", sub_ok);
    snprintf(msg, sizeof msg, "%d corrupted / truncated streams: same status, same bytes when they decode (%d did)", ncor, ncor_dec);
    check(msg, cor_ok);
    free(c); free(a); free(b); free(enc); free(bad);
    return 0;
}

/* the 64 KB content of (key, s) in a chunk buffer: a pattern only it has */
static void fill_sub(uint8_t *chunk, uint64_t key, uint32_t s) {
    for (uint32_t z = 0; z < 16; z++)
        for (uint32_t i = 0; i < 4096; i++) chunk[zshm_sub_off(s, z) + i] = (uint8_t)((key * 31 + s * 7 + z * 13 + i * 5 + (i >> 7)) & 255);
}
static int check_sub(const uint8_t *chunk, uint64_t key, uint32_t s) {
    for (uint32_t z = 0; z < 16; z++)
        for (uint32_t i = 0; i < 4096; i++) if (chunk[zshm_sub_off(s, z) + i] != (uint8_t)((key * 31 + s * 7 + z * 13 + i * 5 + (i >> 7)) & 255)) return 0;
    return 1;
}
static int part2(const char *path) {
    setenv("UFSM_ZCACHE_PATH", path, 1); setenv("UFSM_ZCACHE_GB", "0.004", 1);   /* ~60 slots: constant eviction */
    int nproc = 6; pid_t pid[6];
    for (int k = 0; k < nproc; k++) {
        if ((pid[k] = fork()) == 0) {
            uint8_t *buf = malloc(CV); uint64_t s = 1234 + (uint64_t)k; int wrong = 0, hits = 0;
            if (!zshm_on()) _exit(3);
            for (int t = 0; t < 3000; t++) {
                uint64_t key = lcg(&s) % 200; uint32_t sub = (uint32_t)(lcg(&s) % 32);
                memset(buf, 0, CV);
                if (zshm_get(key * 0x9e3779b97f4a7c15ull, ~key, sub, buf)) { hits++; if (!check_sub(buf, key, sub)) wrong++; }
                else { fill_sub(buf, key, sub); zshm_put(key * 0x9e3779b97f4a7c15ull, ~key, sub, buf); }
            }
            _exit(wrong ? 1 : hits ? 0 : 2);
        }
    }
    int ok = 1;
    for (int k = 0; k < nproc; k++) { int st; waitpid(pid[k], &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) ok = 0; }
    check("zshm: 6 processes x 3000 random get / put on ~60 slots: every hit has its bytes, hits happen", ok);
    if ((pid[0] = fork()) == 0) {   /* a fresh process stores; the parent (which never opened the cache) reads it back */
        uint8_t *buf = malloc(CV); fill_sub(buf, 4242, 5); zshm_put(4242, 77, 5, buf); _exit(zshm_on() ? 0 : 1);
    }
    int st; waitpid(pid[0], &st, 0);
    uint8_t *buf = calloc(1, CV);
    check("zshm: an entry stored by another process is a hit with the same bytes", WIFEXITED(st) && !WEXITSTATUS(st) && zshm_get(4242, 77, 5, buf) && check_sub(buf, 4242, 5));
    free(buf);
    return 0;
}

static int part3(const char *dir, const char *cpath) {
    const int64_t shape[3] = {256, 200, 300}; char arr[600]; snprintf(arr, sizeof arr, "%s/ct.zarr", dir);
    z3w *w = z3w_create(arr, shape, 128, 2.0f, 0, nullptr);
    if (!w) { fprintf(stderr, "z3w: %s\n", z3w_error()); return 1; }
    uint8_t *sb = malloc(CV);
    for (int64_t sz = 0; sz < 2; sz++) for (int64_t sy = 0; sy < 2; sy++) for (int64_t sx = 0; sx < 3; sx++) {
        make_chunk(sb, (int)((sz + sy + sx) % 3), (uint64_t)(sz * 9 + sy * 3 + sx));
        if (z3w_write_shard(w, sz, sy, sx, sb, 1)) { fprintf(stderr, "write: %s\n", z3w_error()); return 1; }
    }
    z3w_close(w); free(sb);
    const int64_t o[3] = {10, 5, 0}, n[3] = {240, 190, 300}; const size_t nv = (size_t)n[0] * n[1] * n[2];
    uint8_t *ref = malloc(nv), *got = malloc(nv);
    char refp[700]; snprintf(refp, sizeof refp, "%s/ref.bin", dir);
    uint32_t masks[4] = {0x00000003u, 0x80000001u, 0x0f0f0f0fu, 0x12345678u};
    pid_t pid = fork();
    if (pid == 0) {   /* the cache off: full read + the submask reads, to a file */
        unsetenv("UFSM_ZCACHE_GB");
        store *st = store_open(dir); z3 *z = z3_open(st, "ct.zarr", nullptr);
        FILE *f = fopen(refp, "wb"); if (!z || !f) _exit(1);
        if (z3_read(z, o, n, ref, 4)) _exit(1);
        fwrite(ref, 1, nv, f);
        for (int k = 0; k < 4; k++) { if (z3_read_sub(z, o, n, ref, 4, masks[k])) _exit(1); fwrite(ref, 1, nv, f); }
        fclose(f); _exit(0);
    }
    int stt; waitpid(pid, &stt, 0);
    if (!WIFEXITED(stt) || WEXITSTATUS(stt)) { check("z3: reference read (cache off)", 0); return 1; }
    setenv("UFSM_ZCACHE_PATH", cpath, 1); setenv("UFSM_ZCACHE_GB", "0.1", 1);
    store *st = store_open(dir); z3 *z = z3_open(st, "ct.zarr", nullptr);
    FILE *f = fopen(refp, "rb");
    int ok = z && f && fread(ref, 1, nv, f) == nv;
    uint64_t h[3], m[3], p[3];
    z3_shared_cache_stats(&h[0], &m[0], &p[0]);
    for (int pass = 0; pass < 2 && ok; pass++) { ok = !z3_read(z, o, n, got, 4) && !memcmp(ref, got, nv); z3_shared_cache_stats(&h[pass + 1], &m[pass + 1], &p[pass + 1]); }   /* cold, then warm */
    char msg[200];
    snprintf(msg, sizeof msg, "z3_read with the shared cache = without it (cold: %llu stored, warm: %llu hits, %llu misses)",
             (unsigned long long)(p[1] - p[0]), (unsigned long long)(h[2] - h[1]), (unsigned long long)(m[2] - m[1]));
    check(msg, ok && p[1] - p[0] == 12 * 32 && h[2] - h[1] == 12 * 32 && m[2] == m[1]);
    int ok2 = 1;
    for (int k = 0; k < 4 && ok2; k++) ok2 = fread(ref, 1, nv, f) == nv && !z3_read_sub(z, o, n, got, 4, masks[k]) && !memcmp(ref, got, nv);
    check("z3_read_sub with the shared cache (4 submasks) = without it", ok2);
    if (f) fclose(f);
    z3_close(z);
    free(ref); free(got);
    return 0;
}

int main(void) {
    char dir[] = "/tmp/test_vfastXXXXXX"; if (!mkdtemp(dir)) return 1;
    char c2[600], c3[600]; snprintf(c2, sizeof c2, "%s/zc2", dir); snprintf(c3, sizeof c3, "%s/zc3", dir);
    part1();
    part2(c2);
    part3(dir, c3);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", dir); if (system(cmd)) {}
    printf("test_volcomp_fast: %s (kernels %s)\n", fails ? "FAIL" : "ok", volcomp_kernels());
    return fails != 0;
}
