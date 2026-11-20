/* ufsm denselab-render: the dense label store (DESIGN "Dense label store (denselab)", 2026-11-19). Every derived label array a sources
   json opens through the derived-array cache (src/dcache.h: the on-the-fly phase per spec and level, the on-the-fly ink, the fibre
   planes once they are cached) rendered whole into ROOT/<key hash>/ (inputs.txt, a/ = a zarr v3 uint8 array, volcomp lossless 128^3
   chunks in S^3 shards, shards.bin = the rendered shards), which the cache then reads instead of computing (dcache_dense_root).
   The values are the cached array's own reads (z3_read of the array the sampler reads), so a rendered cell is byte-identical to the
   on-the-fly one by construction; --verify N checks it against a fresh compute (the array's inner, no cache). */
#include "band_onfly.h"
#include "dcache.h"
#include "fiberraster.h"
#include "inksplat.h"
#include <zstd.h>
#include "sources.h"
#include "store.h"
#include "optcheck.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {   /* (options from argv[3]: argv[2] is SOURCES) */
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static double dl_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int dl_mkdirs(const char *p) {
    char b[1200]; snprintf(b, sizeof b, "%s", p);
    for (char *q = b + 1; *q; q++) if (*q == '/') { *q = 0; if (mkdir(b, 0775) && errno != EEXIST) return -1; *q = '/'; }
    return mkdir(b, 0775) && errno != EEXIST ? -1 : 0;
}
/* one derived array to render: a cached array of a source, its kind / level for the log */
typedef struct { z3 *z; char what[200]; } dl_arr;
static int dl_add(dl_arr **A, int *n, int *cap, z3 *z, const char *what) {
    if (!z) return 0;
    if (dcache_info(z, nullptr, nullptr, nullptr, nullptr)) {   /* not cached itself (a fusion of cached sets, a pooled view): the cached arrays it reads */
        if (!z3_derived_ctx(z)) return 0;
        int r = 0; char w[200]; snprintf(w, sizeof w, "%s (input)", what);
        r += dl_add(A, n, cap, z3_base_of(z), w);
        for (int k = 0; k < 4; k++) { snprintf(w, sizeof w, "%s (input %d)", what, k + 1); r += dl_add(A, n, cap, z3_derived_cover_at(z, k), w); }
        return r;
    }
    char hex[25]; dcache_info(z, nullptr, hex, nullptr, nullptr);
    for (int i = 0; i < *n; i++) { char h2[25]; dcache_info((*A)[i].z, nullptr, h2, nullptr, nullptr); if (!strcmp(h2, hex)) return 0; }   /* (one key: once) */
    if (*n == *cap) { *cap = *cap ? 2 * *cap : 16; *A = realloc(*A, (size_t)*cap * sizeof **A); }
    (*A)[*n].z = z; snprintf((*A)[*n].what, sizeof (*A)[*n].what, "%s", what); (*n)++; return 1;
}
static int dl_kind(const char *kinds, const char *k) {
    const size_t n = strlen(k);
    for (const char *p = kinds; (p = strstr(p, k)); p += n) if ((p == kinds || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1;
    return 0;
}
static int dl_name(const char *list, const char *name) {   /* (nullptr list: every source) */
    if (!list) return 1;
    const size_t n = strlen(name);
    for (const char *p = list; (p = strstr(p, name)); p += n) if ((p == list || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1;
    return 0;
}
static int dl_write_shards(const char *dir, int64_t S, const int64_t g[3], const uint8_t *bits) {
    char p[1200], t[1300]; snprintf(p, sizeof p, "%s/shards.bin", dir); snprintf(t, sizeof t, "%s.%d.tmp", p, (int)getpid());
    FILE *f = fopen(t, "wb"); if (!f) return -1;
    const int64_t h[4] = {S, g[0], g[1], g[2]}; const size_t nb = (size_t)(g[0] * g[1] * g[2]);
    int ok = fwrite("UDL1", 1, 4, f) == 4 && fwrite(h, 8, 4, f) == 4 && fwrite(bits, 1, nb, f) == nb;
    if (fclose(f) || !ok || rename(t, p)) { unlink(t); return -1; }
    return 0;
}

/* render one array: every shard whose box (+ margin) reaches the array's base chunks (z3_box_any), not yet rendered */
typedef struct {
    z3 *z; z3w *w; const char *dir; int64_t S, g[3]; uint8_t *bits; int64_t *todo; size_t ntodo; _Atomic size_t next, done; _Atomic int fail;
    pthread_mutex_t mu; int inner_threads; double t0; uint64_t bytes0; int zprobe; _Atomic uint64_t zb3, zb19;   /* --zstd-probe: the shards' 128^3 chunks as zstd 3 / 19 */
} dl_job;
static void *dl_worker(void *a) {
    dl_job *J = a; const size_t SN = (size_t)J->S * J->S * J->S; uint8_t *buf = malloc(SN);
    if (!buf) { J->fail = 1; return nullptr; }
    for (size_t i; !J->fail && (i = atomic_fetch_add(&J->next, 1)) < J->ntodo;) {
        const int64_t s = J->todo[i], sz = s / (J->g[1] * J->g[2]), sy = s / J->g[2] % J->g[1], sx = s % J->g[2];
        const int64_t o[3] = {sz * J->S, sy * J->S, sx * J->S}, n[3] = {J->S, J->S, J->S};
        if (z3_read(J->z, o, n, buf, J->inner_threads)) { fprintf(stderr, "denselab: %s: read of shard %lld %lld %lld failed\n", J->dir, (long long)sz, (long long)sy, (long long)sx); J->fail = 1; break; }
        if (J->zprobe) {   /* (the codec comparison: what the same chunks cost as the dcache's zstd cells; all-fill chunks omitted, as z3w does) */
            const size_t C3 = 128 * 128 * 128, cap = ZSTD_compressBound(C3); uint8_t *ch = malloc(C3), *zb = malloc(cap); const int fill = z3_meta_of(J->z)->fill;
            for (int64_t cz = 0; cz < J->S; cz += 128) for (int64_t cy = 0; cy < J->S; cy += 128) for (int64_t cx = 0; cx < J->S; cx += 128) {
                int allf = 1;
                for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) { const uint8_t *src = buf + ((size_t)(cz + z) * J->S + (cy + y)) * J->S + cx; memcpy(ch + ((size_t)z * 128 + y) * 128, src, 128);
                                                                             for (int x = 0; x < 128 && allf; x++) allf = src[x] == fill; }
                if (allf) continue;
                size_t r = ZSTD_compress(zb, cap, ch, C3, 3); if (!ZSTD_isError(r)) J->zb3 += r;
                if (J->zprobe > 1) { r = ZSTD_compress(zb, cap, ch, C3, 19); if (!ZSTD_isError(r)) J->zb19 += r; }
            }
            free(ch); free(zb);
        }
        if (z3w_write_shard(J->w, sz, sy, sx, buf, J->inner_threads)) { fprintf(stderr, "denselab: %s: shard write: %s\n", J->dir, z3w_error()); J->fail = 1; break; }
        pthread_mutex_lock(&J->mu); J->bits[s] = 1; const size_t d = ++J->done;
        if (d % 16 == 0 || d == J->ntodo) {
            if (dl_write_shards(J->dir, J->S, J->g, J->bits)) { J->fail = 1; fprintf(stderr, "denselab: %s: cannot write shards.bin\n", J->dir); }
            const double el = dl_now() - J->t0;
            fprintf(stderr, "denselab: %s: %zu of %zu shards, %.0f s (%.1f Mvox/s)\n", J->dir, d, J->ntodo, el, (double)d * SN / 1e6 / (el > 0 ? el : 1));
        }
        pthread_mutex_unlock(&J->mu);
    }
    free(buf); return nullptr;
}
static uint64_t dl_du(const char *dir) {   /* bytes under dir (one level of shard dirs is enough for z3w's layout: walk recursively) */
    char cmd[1400]; snprintf(cmd, sizeof cmd, "du -sb '%s' 2>/dev/null", dir); FILE *p = popen(cmd, "r"); unsigned long long b = 0;
    if (p) { if (fscanf(p, "%llu", &b) != 1) b = 0; pclose(p); }
    return b;
}
static int dl_render(z3 *z, const char *what, const char *root, int64_t S, int S_given, int64_t margin, int nthreads, int dry, size_t limit, int zprobe, double *gsec, uint64_t *gbytes) {
    const char *key; char hex[25]; int C, pd; dcache_info(z, &key, hex, &C, &pd);
    const z3_meta *m = z3_meta_of(z); if (S % C) { fprintf(stderr, "denselab: %s: shard %lld is not a multiple of the cell %d\n", what, (long long)S, C); return 1; }
    char dir[1200], ad[1300]; snprintf(dir, sizeof dir, "%s/%s", root, hex); snprintf(ad, sizeof ad, "%s/a", dir);
    int64_t g[3]; for (int d = 0; d < 3; d++) g[d] = (m->shape[d] + S - 1) / S;
    size_t ns = (size_t)(g[0] * g[1] * g[2]); uint8_t *bits = calloc(ns, 1); int64_t *todo = malloc(ns * sizeof *todo); size_t nt = 0, nskip = 0, nhave = 0;
    { int64_t S2, g2[3]; uint8_t *b2 = nullptr;   /* resume: the shards already rendered */
      if (!dcache_dense_shards(dir, &S2, g2, &b2)) {
          if (S2 != S && !S_given && S2 % C == 0) {   /* (no --shard: the existing store's edge) */
              S = S2; for (int d = 0; d < 3; d++) g[d] = (m->shape[d] + S - 1) / S;
              free(bits); free(todo); ns = (size_t)(g[0] * g[1] * g[2]); bits = calloc(ns, 1); todo = malloc(ns * sizeof *todo); }
          if (S2 == S && g2[0] == g[0] && g2[1] == g[1] && g2[2] == g[2]) memcpy(bits, b2, ns);
          else { fprintf(stderr, "denselab: %s: an existing store with another shard grid: refused (remove it first)\n", dir); free(b2); free(bits); free(todo); return 1; }
          free(b2); } }
    for (int64_t sz = 0; sz < g[0]; sz++) for (int64_t sy = 0; sy < g[1]; sy++) for (int64_t sx = 0; sx < g[2]; sx++) {
        const int64_t s = (sz * g[1] + sy) * g[2] + sx; if (bits[s]) { nhave++; continue; }
        const int64_t o[3] = {sz * S - margin, sy * S - margin, sx * S - margin}, n[3] = {S + 2 * margin, S + 2 * margin, S + 2 * margin};
        const int any = z3_box_any(z, o, n); if (any < 0) { fprintf(stderr, "denselab: %s: box test failed\n", what); free(bits); free(todo); return 1; }
        if (any) todo[nt++] = s; else nskip++;
    }
    printf("denselab: %s -> %s (cell %d%s): shape %lld x %lld x %lld, %zu shards of %lld^3: %zu to render, %zu already, %zu out of the labels' reach\n",
           what, dir, C, pd ? ", near-lossless" : "", (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], ns, (long long)S, nt, nhave, nskip);
    fflush(stdout);
    if (dry) { free(bits); free(todo); return 0; }
    const int partial = limit && nt > limit;   /* --limit N: an evenly spread sample of the shards (sizes / speed); the store stays partial */
    if (partial) { for (size_t i = 0; i < limit; i++) todo[i] = todo[(size_t)((double)i * nt / limit)]; nt = limit; }
    if (dl_mkdirs(dir)) { fprintf(stderr, "denselab: cannot create %s\n", dir); free(bits); free(todo); return 1; }
    { char ip[1300]; snprintf(ip, sizeof ip, "%s/inputs.txt", dir); FILE *f = fopen(ip, "wb"); if (!f || fprintf(f, "%s\n", key) < 0 || fclose(f)) { fprintf(stderr, "denselab: cannot write %s\n", ip); free(bits); free(todo); return 1; } }
    /* the shards not in reach are rendered too in the bits (their values are the fill: the cache computes nothing for them either way,
       but a reader then never computes them): marked without a write (z3w omits all-fill chunks) */
    char attrs[1600]; snprintf(attrs, sizeof attrs, "{\"denselab\": {\"key_hash\": \"%s\", \"cell\": %d}}", hex, C);
    z3w *w = z3w_create(ad, m->shape, (int)S, 0, m->fill, attrs); if (!w) { fprintf(stderr, "denselab: %s: %s\n", ad, z3w_error()); free(bits); free(todo); return 1; }
    const double t0 = dl_now();
    dl_job J = {.z = z, .w = w, .dir = dir, .S = S, .g = {g[0], g[1], g[2]}, .bits = bits, .todo = todo, .ntodo = nt, .t0 = t0, .zprobe = zprobe};
    pthread_mutex_init(&J.mu, nullptr);
    const int nw = nthreads < 4 ? 1 : nthreads / 4; J.inner_threads = nthreads / nw > 0 ? nthreads / nw : 1;   /* (a few shards at once, each read with the rest) */
    pthread_t th[64]; const int m2 = nw > 64 ? 64 : nw;
    for (int t = 0; t < m2; t++) pthread_create(&th[t], nullptr, dl_worker, &J);
    for (int t = 0; t < m2; t++) pthread_join(th[t], nullptr);
    int rc = J.fail || z3w_close(w) ? 1 : 0;
    if (!rc && !partial) { for (size_t i = 0; i < ns; i++) bits[i] = 1; rc = dl_write_shards(dir, S, g, bits) ? 1 : 0; }   /* (complete: the out-of-reach shards are fill) */
    const double el = dl_now() - t0; const uint64_t by = dl_du(ad);
    printf("denselab: %s: %s, %zu shards in %.0f s, store %.3f GB (%.3f bit per voxel of the rendered shards)\n", what, rc ? "FAILED" : "done", nt, el, by / 1e9,
           nt ? 8.0 * by / ((double)nt * S * S * S) : 0);
    if (zprobe) printf("denselab: %s: codec probe over the rendered shards: volcomp lossless %.4f GB, zstd 3 %.4f GB%s%.4f GB\n", what, by / 1e9, J.zb3 / 1e9,
                       zprobe > 1 ? ", zstd 19 " : "", zprobe > 1 ? J.zb19 / 1e9 : 0.);
    *gsec += el; *gbytes += by;
    if (!rc && !partial) { char dp[1300]; snprintf(dp, sizeof dp, "%s/done", dir); FILE *f = fopen(dp, "wb"); if (f) { fprintf(f, "shards %zu rendered %zu bytes %llu seconds %.0f\n", ns, nt, (unsigned long long)by, el); fclose(f); } }
    pthread_mutex_destroy(&J.mu); free(bits); free(todo); return rc;
}
/* --verify N: N random boxes per array inside its rendered shards: the cached read (served by the store; no local cache in this process)
   vs the array's inner (computed now): byte for byte */
static int dl_verify(z3 *z, const char *what, int nbox, int64_t P, uint64_t seed, int nthreads) {
    char hex[25]; dcache_info(z, nullptr, hex, nullptr, nullptr); const char *R = dcache_dense_root();
    char dir[1200]; snprintf(dir, sizeof dir, "%s/%s", R ? R : "", hex);
    int64_t S, g[3]; uint8_t *bits = nullptr;
    if (!R || dcache_dense_shards(dir, &S, g, &bits)) { printf("denselab verify: %s: no store at %s\n", what, dir); return 1; }
    z3 *in = z3_base_of(z); const z3_meta *m = z3_meta_of(z); const size_t N = (size_t)P * P * P;
    uint8_t *a = malloc(N), *b = malloc(N); int bad = 0, tried = 0; size_t nonfill = 0;
    dcache_stats s0; dcache_get_stats(&s0);
    for (int k = 0; k < nbox * 50 && tried < nbox && !bad; k++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull; int64_t o[3];
        for (int d = 0; d < 3; d++) { const uint64_t r = (seed >> (11 + 7 * d)) & 0x3fffff; o[d] = m->shape[d] > P ? (int64_t)(r % (uint64_t)(m->shape[d] - P)) : 0; }
        const int64_t cs = ((o[0] / S) * g[1] + o[1] / S) * g[2] + o[2] / S; if (!bits[cs]) continue;
        const int64_t n[3] = {P, P, P};
        if (k < nbox * 25) { const int any = z3_box_any(z, o, n); if (any == 0) continue; }   /* (prefer boxes with labels; later any) */
        tried++;
        if (z3_read(z, o, n, a, nthreads) || z3_read(in, o, n, b, nthreads)) { printf("denselab verify: %s: read failed\n", what); bad = 1; break; }
        size_t nd = 0; for (size_t i = 0; i < N; i++) { nd += a[i] != b[i]; nonfill += b[i] != m->fill; }
        if (nd) { printf("denselab verify: %s: box %lld %lld %lld: %zu of %zu voxels differ: FAIL\n", what, (long long)o[0], (long long)o[1], (long long)o[2], nd, N); bad = 1; }
    }
    dcache_stats s1; dcache_get_stats(&s1);
    printf("denselab verify: %s: %d boxes of %lld^3, %zu non-fill voxels, %llu cells from the store, %llu computed: %s\n", what, tried, (long long)P, nonfill,
           (unsigned long long)(s1.cells_dense - s0.cells_dense), (unsigned long long)(s1.cells_computed - s0.cells_computed), bad ? "FAIL" : tried ? "byte-identical" : "nothing checked");
    free(a); free(b); free(bits); return bad || !tried || s1.cells_dense == s0.cells_dense;
}
int cmd_denselab_render(int argc, char **argv) {
    if (argc < 3 || argv[2][0] == '-') {
        fprintf(stderr, "usage: ufsm denselab-render SOURCES.json --out ROOT [--source N1,N2] [--kind phase,ink,fibre] [--levels a,b] [--shard 512] [--margin 128]\n"
                        "                             [--threads 16] [--gpu D --gpu-slots 2 --gpu-mb 1024] [--dry 1] [--limit N] [--zstd-probe 1|2] [--verify N --P 96 --seed 1]\n"
                        "  renders every derived (cached) label array of the sources into ROOT/<key hash>/ (zarr + shards.bin); the derived cache\n"
                        "  then reads them (with UFSM_DENSELAB=ROOT; unset = off). --verify N: N boxes per array, the store vs a fresh compute.\n");
        return 2;
    }
    const char *out = opt(argc, argv, "--out", nullptr); if (!out) { fprintf(stderr, "denselab-render: --out ROOT\n"); return 2; }
    const int nverify = opt_atoi("--verify", opt(argc, argv, "--verify", "0"));
    /* render: the store is written, never read (no dense tier), and the local disk cache is not filled (a whole scroll would flood it);
       verify: the store read, no local disk cache (every cell from the store or computed) */
    setenv("UFSM_DENSELAB", nverify ? out : "off", 1);
    if (!opt(argc, argv, "--local-cache", nullptr) || nverify) setenv("UFSM_DERIVED_CACHE", "off", 1);
    const char *names = opt(argc, argv, "--source", nullptr), *kinds = opt(argc, argv, "--kind", "phase,ink,fibre"), *lv = opt(argc, argv, "--levels", nullptr);
    const int64_t S = opt_atoi("--shard", opt(argc, argv, "--shard", "512")), margin = opt_atoi("--margin", opt(argc, argv, "--margin", "128"));
    const int nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 16), dry = opt_atoi("--dry", opt(argc, argv, "--dry", "0"));
    if (S < 128 || S % 128) { fprintf(stderr, "denselab-render: --shard a multiple of 128\n"); return 2; }
    int la = 0, lb = MAXLEV - 1; if (lv && sscanf(lv, "%d,%d", &la, &lb) < 1) { fprintf(stderr, "denselab-render: --levels a,b\n"); return 2; }
    if (lv && !strchr(lv, ',')) lb = la;
    sources *SS = sources_load(argv[2]); if (!SS) { fprintf(stderr, "denselab-render: cannot load %s\n", argv[2]); return 1; }
    const int gpu = opt_atoi("--gpu", opt(argc, argv, "--gpu", "-1"));
    if (gpu >= 0) {   /* the training kernels (byte-identical to train's --onfly-gpu path): the pair rule (--gpu-slots tiles of --gpu-mb), ink, fibre */
        const int gs = opt_atoi("--gpu-slots", opt(argc, argv, "--gpu-slots", "2")), gm = opt_atoi("--gpu-mb", opt(argc, argv, "--gpu-mb", "1024"));
        if (phase_onfly_config(gpu, gs, gm, 1)) { fprintf(stderr, "denselab-render: --gpu %d: no GPU pair rule (or the reserve failed)\n", gpu); sources_free(SS); return 1; }
        inkfly_gpu_config(gpu);
        int nfib = 0; for (int i = 0; i < SS->n; i++) for (int c = 0; c < NTG; c++) nfib += SS->src[i].tgt_fib[c] && dl_name(names, SS->src[i].name);
        if (nfib && dl_kind(kinds, "fibre") && fiber_onfly_config(gpu, (int)S) < 0) fprintf(stderr, "denselab-render: --gpu %d: fibre planes on the CPU\n", gpu);
        printf("denselab-render: on-the-fly labels on GPU %d (pair rule %d slots of %d MB)\n", gpu, gs, gm);
    }
    dl_arr *A = nullptr; int na = 0, ca = 0, nsel = 0;
    for (int i = 0; i < SS->n; i++) {
        source *s = &SS->src[i]; if (!dl_name(names, s->name)) continue; nsel++;
        const int l0 = s->min_level > la ? s->min_level : la, l1 = s->max_level >= 0 && s->max_level < lb ? s->max_level : lb;
        for (int l = l0; l <= l1 && l < MAXLEV; l++) {
            char w[160];
            if (dl_kind(kinds, "phase")) { snprintf(w, sizeof w, "%s phase L%d", s->name, l); dl_add(&A, &na, &ca, source_phase_at(s, l, 0), w); }
            if (dl_kind(kinds, "ink")) { int st; snprintf(w, sizeof w, "%s ink L%d", s->name, l); z3 *t = source_tgt_for_level(s, CH_SHEET, l, &st); dl_add(&A, &na, &ca, t, w);
                                         if (s->tgt_val[CH_SHEET][l]) { snprintf(w, sizeof w, "%s ink-val L%d", s->name, l); dl_add(&A, &na, &ca, s->tgt_val[CH_SHEET][l], w); } }
            if (dl_kind(kinds, "fibre")) for (int ch = CH_FIB; ch <= CH_FIBV; ch++) { int st; snprintf(w, sizeof w, "%s %s L%d", s->name, ch == CH_FIB ? "fibre" : ch == CH_FIBH ? "fibre_h" : "fibre_v", l);
                                                                                    dl_add(&A, &na, &ca, source_tgt_for_level(s, ch, l, &st), w); }
        }
    }
    printf("denselab-render: %d sources selected, %d cached derived arrays (%s)\n", nsel, na, nverify ? "verify" : dry ? "dry run" : "render");
    int rc = 0; double sec = 0; uint64_t by = 0;
    for (int i = 0; i < na; i++)
        rc |= nverify ? dl_verify(A[i].z, A[i].what, nverify, opt_atoi("--P", opt(argc, argv, "--P", "96")), (uint64_t)opt_atoi("--seed", opt(argc, argv, "--seed", "1")) + 7919u * (unsigned)i, nt)
                      : dl_render(A[i].z, A[i].what, out, S, opt(argc, argv, "--shard", nullptr) != nullptr, margin, nt, dry, (size_t)opt_atoi("--limit", opt(argc, argv, "--limit", "0")), opt_atoi("--zstd-probe", opt(argc, argv, "--zstd-probe", "0")), &sec, &by);
    if (!nverify && !dry) printf("denselab-render: %d arrays, %.3f GB, %.0f s%s\n", na, by / 1e9, sec, rc ? ", FAILED" : "");
    free(A); sources_free(SS);
    return rc;
}
/* ufsm ink-rayface (2026-11-20): every onfly ink source's MIL face-drop bits (inksplat.h inkfly_rayface_build) into its cache dir, so the
   sampler's MIL mode 1 reads no CT (sampler-only P 256, the ink ladder's sources: 18.1 -> ~52 Mvox/s). --verify N: N windows per source,
   the rays from the bits vs the face search, byte for byte */
int cmd_ink_rayface(int argc, char **argv) {
    if (argc < 3 || argv[2][0] == '-') {
        fprintf(stderr, "usage: ufsm ink-rayface SOURCES.json [--source N1,N2] [--threads 8] [--gpu D] [--force 1] [--rays 2] [--verify N --P 256 --stride 2 --R 10 --seed 1]\n"
                        "  --rays S: also the per-tile ray lists of MIL mode 1 at stride S (0: none); --verify compares the rays a window gets from them / the bits\n"
                        "  with the face search on the CT; --diag N: the CT profiles of N faceless pixels per source, classified (threshold vs mesh offset)\n"
                        "  precomputes the MIL rays' face-drop bit of every supervised label pixel of each onfly ink target (rayface.bin in its cache dir)\n");
        return 2;
    }
    const char *names = opt(argc, argv, "--source", nullptr);
    const int nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 8), gpu = opt_atoi("--gpu", opt(argc, argv, "--gpu", "-1"));
    const int force = opt_atoi("--force", opt(argc, argv, "--force", "0")), nver = opt_atoi("--verify", opt(argc, argv, "--verify", "0"));
    const int rays = opt_atoi("--rays", opt(argc, argv, "--rays", "2")), diag = opt_atoi("--diag", opt(argc, argv, "--diag", "0"));
    const int P = opt_atoi("--P", opt(argc, argv, "--P", "256")), stride = opt_atoi("--stride", opt(argc, argv, "--stride", "2"));
    const float R = (float)atof(opt(argc, argv, "--R", "10")); uint64_t seed = (uint64_t)opt_atoi("--seed", opt(argc, argv, "--seed", "1"));
    if (nver) setenv("UFSM_INK_RAYFACE", "0", 1);   /* (verify: the sources open without the bits; the build loads them per handle below) */
    sources *SS = sources_load(argv[2]); if (!SS) { fprintf(stderr, "ink-rayface: cannot load %s\n", argv[2]); return 1; }
    if (nver) unsetenv("UFSM_INK_RAYFACE");
    inkfly_gpu_config(gpu);   /* (-1: the CPU face search, none tried on a device) */
    int rc = 0, nh = 0;
    for (int i = 0; i < SS->n && !rc; i++) {
        source *s = &SS->src[i]; if (!dl_name(names, s->name)) continue;
        for (int c = 0; c < NTG && !rc; c++) {
            void *h = s->ink_fly[c]; if (!h) continue; nh++;
            const double t0 = dl_now();
            if (inkfly_rayface_build(h, nt, force || nver)) { fprintf(stderr, "ink-rayface: %s: FAILED\n", s->name); rc = 1; break; }
            printf("ink-rayface: %s: built in %.0f s\n", s->name, dl_now() - t0);
            if (rays > 0) {   /* the per-tile ray lists of MIL mode 1 at --rays S (the trainer's --ink-mil-stride, 2) */
                const double t1 = dl_now();
                if (inkfly_raylist_build(h, rays, 1, nt)) { fprintf(stderr, "ink-rayface: %s: ray lists FAILED\n", s->name); rc = 1; break; }
                printf("ink-rayface: %s: ray lists (stride %d, mode 1) in %.0f s\n", s->name, rays, dl_now() - t1);
            }
            if (diag > 0 && inkfly_face_diag(h, diag, seed)) { rc = 1; break; }
            if (!nver) continue;
            /* windows at the onfly level around random supervised tiles: rays from the bits vs rays with the bits off (the face search) */
            int ok = 0, tried = 0; size_t nrays = 0;
            for (int k = 0; k < nver * 20 && tried < nver; k++) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                int64_t o[3]; if (inkfly_sample_window(h, seed, P, o)) continue;
                const int64_t n[3] = {P, P, P}; inkray *a = nullptr, *b = nullptr; size_t na = 0, nb = 0;
                if (inkfly_rays(h, o, n, 1, stride, R, &a, &na) || inkfly_rays_noface(h, o, n, 1, stride, R, &b, &nb)) { free(a); free(b); rc = 1; break; }
                tried++; nrays += na;
                const int same = na == nb && (!na || !memcmp(a, b, na * sizeof *a));
                if (!same) printf("ink-rayface verify: %s: window %lld %lld %lld: %zu rays from the bits, %zu from the face search: DIFFER\n", s->name, (long long)o[0], (long long)o[1], (long long)o[2], na, nb);
                ok += same; free(a); free(b);
            }
            printf("ink-rayface verify: %s: %d of %d windows (P %d, stride %d) byte-identical, %zu rays%s\n", s->name, ok, tried, P, stride, nrays, ok == tried && tried ? "" : "  FAIL");
            if (ok != tried || !tried) rc = 1;
        }
    }
    if (!nh) { fprintf(stderr, "ink-rayface: no onfly ink target in the selected sources\n"); rc = 1; }
    sources_free(SS); return rc;
}
