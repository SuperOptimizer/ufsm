/* gpudec (src/nn_gpudec.cu): volcomp substream decoding on the GPU = the reference vf_decode_sub, byte for byte.
   1. synthetic lossy chunks (smooth, noisy, flat + busy blocks, edges / clipping) at q 1..80: every substream, random masks;
      16 threads at once (the service batches them).
   2. corrupted / truncated streams that still parse: the GPU decodes a request (0) exactly when the reference decodes every
      substream in it, with its bytes; else it reports 1 (the caller then decodes on the CPU), and never touches dst.
   3. z3_read_sub of a lossy zarr with the GPU on = the reference decode of the raw chunks (z3_chunk_raw), random submasks.
   Real data / speed: test_gpudec --real ARRAY cz0 cy0 cx0 ncz ncy ncx [threads] [submask-density]
      (ARRAY = a zarr array dir, e.g. .../ct.zarr/0): loads the raw chunks of the box, decodes them on the CPU (volcomp_fast.h,
      threads) and on the GPU (threads submitting at once), compares every byte against vf_decode_sub and prints MB/s. */
#include "gpudec.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#define VOLCOMP_IMPLEMENTATION
#include "volcomp.h"
#include "volcomp_fast.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define CV VOLCOMP_CHUNK_VOXELS
static uint64_t lcg(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 17; }
static int fails = 0;
static void check(const char *what, int ok) { printf("gpudec: %-100s %s\n", what, ok ? "ok" : "FAIL"); fails += !ok; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void make_chunk(uint8_t *c, int kind, uint64_t seed) {
    uint64_t s = seed;
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
        size_t i = ((size_t)z * 128 + y) * 128 + x; int v;
        switch (kind) {
        case 0: v = 128 + (int)(60 * __builtin_sinf(x * 0.11f + y * 0.07f) * __builtin_cosf(z * 0.05f)) + (int)(lcg(&s) % 9) - 4; break;
        case 1: v = (int)(lcg(&s) & 255); break;
        case 2: v = ((x >> 4) + (y >> 4) + (z >> 4)) % 3 ? 40 : 40 + (int)(lcg(&s) % 200); break;
        default: v = x < 64 ? 0 : (y > 90 ? 255 : 100 + (z & 31)); break;
        }
        c[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}
static void to_gd(const vf_parsed *p, gd_chunk *g) {
    *g = (gd_chunk){p->payload, p->tok_n, p->byp_n, p->off, p->step, {0}, VF_SCAN16};
    for (int m = 0; m < 10; m++) g->dt[m] = (const uint32_t *)p->models[m].dt;
}
/* the reference: vf_decode_sub for every s in mask into a zeroed chunk; status of the first failure */
static volcomp_status ref_subs(const vf_parsed *p, uint32_t mask, uint8_t *dst) {
    memset(dst, 0, CV);
    for (uint32_t s = 0; s < VF_NSUB; s++) if (mask >> s & 1) { volcomp_status st = vf_decode_sub(p, s, dst, -1, NULL); if (st != VOLCOMP_OK) return st; }
    return VOLCOMP_OK;
}

/* ---- 1 + 2: encoded synthetic chunks ---- */
enum { NENC = 24 };
static uint8_t *g_enc[NENC]; static size_t g_n[NENC];
typedef struct { atomic_int bad, gpu, seed; } mt_arg;
static void *mt_worker(void *a) {
    mt_arg *A = a; uint8_t *r = malloc(CV), *g = malloc(CV); uint64_t s = 777 + (uint64_t)atomic_fetch_add(&A->seed, 1);
    for (int it = 0; it < 24; it++) {
        int k = (int)(lcg(&s) % NENC); vf_parsed p;
        if (vf_parse(g_enc[k], g_n[k], &p) != VOLCOMP_OK) continue;
        uint32_t mask = it % 3 == 0 ? 0xffffffffu : (uint32_t)lcg(&s) ^ (uint32_t)(lcg(&s) << 15);
        gd_chunk gc; to_gd(&p, &gc); ref_subs(&p, mask, r); memset(g, 0, CV);
        int rc = gpudec_decode(&gc, mask, g);
        if (rc == 0) atomic_fetch_add(&A->gpu, 1);
        if (rc != 0 || memcmp(r, g, CV)) atomic_fetch_add(&A->bad, 1);
    }
    free(r); free(g); return nullptr;
}
static void part12(void) {
    uint8_t *c = malloc(CV), *a = malloc(CV), *b = malloc(CV), *bad = malloc(VOLCOMP_ENCODE_BOUND);
    const float qs[] = {1, 2, 4, 8, 20, 80};
    int ne = 0, all_ok = 1, nall = 0; uint64_t s = 99;
    for (int kind = 0; kind < 4; kind++)
        for (int qi = 0; qi < 6; qi++) {
            make_chunk(c, kind, 1000 + kind * 10 + qi);
            uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND); size_t n = 0;
            if (volcomp_encode(c, qs[qi], enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK || enc[5] != VLL_MODE_LOSSY) { free(enc); continue; }
            g_enc[ne] = enc; g_n[ne++] = n;
            vf_parsed p; if (vf_parse(enc, n, &p) != VOLCOMP_OK) { all_ok = 0; continue; }
            gd_chunk gc; to_gd(&p, &gc);
            for (int t = 0; t < 6; t++) {
                uint32_t mask = t == 0 ? 0xffffffffu : t == 1 ? 1u << (lcg(&s) & 31) : (uint32_t)lcg(&s) ^ (uint32_t)(lcg(&s) << 15);
                ref_subs(&p, mask, a); memset(b, 0, CV);
                if (gpudec_decode(&gc, mask, b) != 0 || memcmp(a, b, CV)) all_ok = 0;
                nall++;
            }
        }
    char msg[200];
    snprintf(msg, sizeof msg, "%d synthetic chunks (4 kinds x q 1..80), %d requests (all / one / random substreams) = vf_decode_sub", ne, nall);
    check(msg, ne > 12 && all_ok);
    mt_arg A; memset(&A, 0, sizeof A); pthread_t th[16];
    for (int t = 0; t < 16; t++) pthread_create(&th[t], nullptr, mt_worker, &A);
    for (int t = 0; t < 16; t++) pthread_join(th[t], nullptr);
    uint64_t subs, batches, redo; double gs; gpudec_stats(&subs, &batches, &redo, &gs);
    snprintf(msg, sizeof msg, "16 threads x 24 requests at once: all on the GPU, = vf_decode_sub (%llu substreams in %llu batches so far)",
             (unsigned long long)subs, (unsigned long long)batches);
    check(msg, atomic_load(&A.bad) == 0 && atomic_load(&A.gpu) == 16 * 24);

    /* 2: corrupted copies that still parse */
    int ncor = 0, nref_ok = 0, ngpu_ok = 0, cor_ok = 1;
    for (int k = 0; k < ne; k++)
        for (int t = 0; t < 60; t++) {
            size_t m = g_n[k]; memcpy(bad, g_enc[k], m);
            if (t % 6 == 5) m = 8 + lcg(&s) % (m - 8);
            else for (int f = 0; f < 1 + t % 3; f++) bad[8 + lcg(&s) % (m - 8)] ^= (uint8_t)(1u << (lcg(&s) & 7));
            vf_parsed p; if (vf_parse(bad, m, &p) != VOLCOMP_OK) continue;
            uint32_t mask = t % 2 ? 0xffffffffu : (uint32_t)lcg(&s) ^ (uint32_t)(lcg(&s) << 15);
            gd_chunk gc; to_gd(&p, &gc);
            volcomp_status rs = ref_subs(&p, mask, a);
            memset(b, 0x5a, CV);
            int rc = gpudec_decode(&gc, mask, b);
            ncor++; nref_ok += rs == VOLCOMP_OK; ngpu_ok += rc == 0;
            if (rs == VOLCOMP_OK) { for (size_t i = 0; i < CV; i++) if (!(mask >> ((i >> 18) * 4 + ((i >> 12) & 3)) & 1)) a[i] = 0x5a; }
            if (rs == VOLCOMP_OK ? rc != 0 || memcmp(a, b, CV) : rc != 1) cor_ok = 0;
            if (rs != VOLCOMP_OK) { int touched = 0; for (size_t i = 0; i < CV && !touched; i++) touched = b[i] != 0x5a; if (touched) cor_ok = 0; }
        }
    snprintf(msg, sizeof msg, "%d corrupted / truncated streams that parse: GPU ok exactly when the reference is (%d), same bytes, else 1",
             ncor, nref_ok);
    check(msg, ncor > 200 && cor_ok && ngpu_ok == nref_ok);
    free(c); free(a); free(b); free(bad);
}

/* ---- 3: z3_read_sub on the GPU = the reference decode of the raw chunks ---- */
static void part3(const char *dir) {
    const int64_t shape[3] = {256, 256, 384}; char arr[600]; snprintf(arr, sizeof arr, "%s/ct.zarr", dir);
    z3w *w = z3w_create(arr, shape, 256, 2.0f, 0, nullptr);
    if (!w) { check("z3w_create", 0); return; }
    uint8_t *vol = malloc((size_t)256 * 256 * 256);
    for (int sx = 0; sx < 2; sx++) {
        for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++)
            vol[((size_t)z * 256 + y) * 256 + x] = (uint8_t)(120 + 50 * __builtin_sinf((x + 256 * sx) * 0.09f + y * 0.05f) * __builtin_cosf(z * 0.04f) + ((x * 7 + y * 13 + z * 3) & 7));
        if (z3w_write_shard(w, 0, 0, sx, vol, 4)) { check("z3w_write_shard", 0); return; }
    }
    z3w_close(w); free(vol);
    store *st = store_open(dir); z3 *z = z3_open(st, "ct.zarr", nullptr);
    if (!z) { check("z3_open", 0); return; }
    uint8_t *got = malloc(CV), *ref = malloc(CV); int ok = 1, n = 0; uint64_t s = 5;
    uint64_t s0; gpudec_stats(&s0, nullptr, nullptr, nullptr);
    for (int cz = 0; cz < 2; cz++) for (int cy = 0; cy < 2; cy++) for (int cx = 0; cx < 3; cx++)
        for (int t = 0; t < 3; t++) {
            uint8_t *raw; size_t rn; vf_parsed p;
            if (z3_chunk_raw(z, cz, cy, cx, &raw, &rn) != 1 || vf_parse(raw, rn, &p) != VOLCOMP_OK) { ok = 0; continue; }
            const uint32_t mask = t == 0 ? 0xffffffffu : (uint32_t)lcg(&s) | 1u;
            ref_subs(&p, mask, ref); free(raw);
            const int64_t o[3] = {cz * 128, cy * 128, cx * 128}, nn[3] = {128, 128, 128};
            if (z3_read_sub(z, o, nn, got, 2, mask) || memcmp(got, ref, CV)) ok = 0;
            n++;
        }
    uint64_t s1; gpudec_stats(&s1, nullptr, nullptr, nullptr);
    char msg[200]; snprintf(msg, sizeof msg, "z3_read_sub (%d chunk reads, full + random submasks) = reference, %llu substreams on the GPU", n, (unsigned long long)(s1 - s0));
    check(msg, ok && s1 > s0);
    z3_close(z); store_close(st); free(got); free(ref);
}

/* ---- --real: raw chunks of a real array, CPU vs GPU ---- */
typedef struct { uint8_t **enc; size_t *n; uint32_t *mask; int nc; _Atomic int next, bad, gpu, cpu; int mode; } real_job;   /* mode 0 CPU (vfx), 1 GPU, 2 check */
static void *real_worker(void *a) {
    real_job *J = a; uint8_t *d = malloc(CV), *r = J->mode == 2 ? malloc(CV) : nullptr;
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < J->nc;) {
        vf_parsed p; if (vf_parse(J->enc[k], J->n[k], &p) != VOLCOMP_OK) { atomic_fetch_add(&J->bad, 1); continue; }
        if (J->mode == 0) { if (vfx_decode_subs(&p, J->mask[k], d) != VOLCOMP_OK) atomic_fetch_add(&J->bad, 1); continue; }
        gd_chunk gc; to_gd(&p, &gc);
        if (J->mode == 2) memset(d, 0, CV);
        int rc = gpudec_decode(&gc, J->mask[k], d);
        if (rc == 0) atomic_fetch_add(&J->gpu, 1); else { atomic_fetch_add(&J->cpu, 1); vfx_decode_subs(&p, J->mask[k], d); }
        if (J->mode == 2 && (ref_subs(&p, J->mask[k], r) != VOLCOMP_OK || memcmp(r, d, CV))) atomic_fetch_add(&J->bad, 1);
    }
    free(d); free(r); return nullptr;
}
static double g_cpu_s;   /* process CPU seconds of the last real_run */
static double cpu_now(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) * 1e-6; }
static double real_run(real_job *J, int mode, int nt) {
    J->next = 0; J->bad = J->gpu = J->cpu = 0; J->mode = mode; pthread_t th[256];
    const double t0 = now_s(), c0 = cpu_now();
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, real_worker, J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    g_cpu_s = cpu_now() - c0;
    return now_s() - t0;
}
static int real_main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: test_gpudec --real ARRAY cz0 cy0 cx0 ncz ncy ncx [threads] [submask density 0..1, default 1]\n"); return 2; }
    char root[1024]; snprintf(root, sizeof root, "%s", argv[2]); char *sl = strrchr(root, '/'); if (!sl) return 2; *sl = 0;
    store *st = store_open(root); z3 *z = z3_open(st, sl + 1, nullptr);
    if (!z) { fprintf(stderr, "z3_open: %s\n", z3_error()); return 1; }
    const int64_t c0[3] = {atoll(argv[3]), atoll(argv[4]), atoll(argv[5])}, cn[3] = {atoll(argv[6]), atoll(argv[7]), atoll(argv[8])};
    const int nt = argc > 9 ? atoi(argv[9]) : 8; const double dens = argc > 10 ? atof(argv[10]) : 1.0;
    const int cap = (int)(cn[0] * cn[1] * cn[2]);
    real_job J = {0}; J.enc = calloc(cap, sizeof *J.enc); J.n = calloc(cap, sizeof *J.n); J.mask = calloc(cap, sizeof *J.mask);
    size_t enc_bytes = 0; uint64_t subs = 0, s = 1;
    for (int64_t a = 0; a < cn[0]; a++) for (int64_t b = 0; b < cn[1]; b++) for (int64_t c = 0; c < cn[2]; c++) {
        uint8_t *raw; size_t rn;
        if (z3_chunk_raw(z, c0[0] + a, c0[1] + b, c0[2] + c, &raw, &rn) != 1) continue;
        uint32_t mode; if (vll_chunk_mode(raw, rn, &mode) != VOLCOMP_OK || mode != VLL_MODE_LOSSY) { free(raw); continue; }
        uint32_t m = 0; for (int q = 0; q < 32; q++) if (dens >= 1 || (lcg(&s) % 10000) < dens * 10000) m |= 1u << q;
        if (!m) m = 1;
        J.enc[J.nc] = raw; J.n[J.nc] = rn; J.mask[J.nc++] = m; enc_bytes += rn; subs += (uint64_t)__builtin_popcount(m);
    }
    const double mb = subs * 65536.0 / 1e6;
    printf("gpudec --real: %d lossy chunks, %.1f MB encoded, %llu substreams = %.0f MB decoded, %d threads\n", J.nc, enc_bytes / 1e6, (unsigned long long)subs, mb, nt);
    double tc = real_run(&J, 2, nt);
    printf("  check vs vf_decode_sub: %s (%d chunks on the GPU, %d on the CPU, %d mismatches) %.2f s\n", J.bad ? "FAIL" : "identical", J.gpu, J.cpu, J.bad, tc);
    int bad = J.bad;
    double t_cpu = real_run(&J, 0, nt);
    printf("  CPU volcomp_fast, %d threads: %.2f s = %.0f MB/s (%.0f MB/s per thread), %.1f CPU-s\n", nt, t_cpu, mb / t_cpu, mb / t_cpu / nt, g_cpu_s);
    uint64_t s0, b0; double g0; gpudec_stats(&s0, &b0, nullptr, &g0);
    double t_gpu = real_run(&J, 1, nt);
    uint64_t s1, b1; double g1; gpudec_stats(&s1, &b1, nullptr, &g1);
    printf("  GPU, %d submitting threads: %.2f s = %.0f MB/s (%llu batches, %.0f substreams / batch, GPU busy %.2f s, %.1f CPU-s, %d chunks on the CPU)\n", nt, t_gpu, mb / t_gpu,
           (unsigned long long)(b1 - b0), (double)(s1 - s0) / (double)(b1 - b0 ? b1 - b0 : 1), g1 - g0, g_cpu_s, J.cpu);
    return bad != 0;
}

int main(int argc, char **argv) {
    setenv("UFSM_GPU_DECODE", "1", 0);
    if (!(argc > 1 && !strcmp(argv[1], "--real"))) setenv("UFSM_GPU_DECODE_MIN_BYTES", "0", 1);   /* the tests: every request on the GPU */
    if (!gpudec_enabled()) { printf("test_gpudec: no GPU decoder (no CUDA device): skipped\n"); return 0; }
    if (argc > 1 && !strcmp(argv[1], "--real")) return real_main(argc, argv);
    char dir[] = "/tmp/test_gpudecXXXXXX"; if (!mkdtemp(dir)) return 1;
    part12();
    part3(dir);
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", dir); if (system(cmd)) {}
    uint64_t subs, batches, redo; double gs; gpudec_stats(&subs, &batches, &redo, &gs);
    printf("test_gpudec: %s (%llu substreams, %llu batches, %llu requests redone on the CPU, GPU %.2f s)\n", fails ? "FAIL" : "ok",
           (unsigned long long)subs, (unsigned long long)batches, (unsigned long long)redo, gs);
    return fails != 0;
}
