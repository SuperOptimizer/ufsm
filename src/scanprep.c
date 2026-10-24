/* ufsm mirror / ufsm scanprep: see src/scanprep.h. */
#include "scanprep.h"
#include "edt.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

uint32_t crc32c(uint32_t crc, const void *buf, size_t len);   /* src/z3w.c */

static const char *sopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static double snow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int smkdirs(const char *path) {
    char p[1400]; snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(p, 0755) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(p, 0755) && errno != EEXIST ? -1 : 0;
}
static int write_file(const char *path, const void *buf, size_t n) {   /* tmp + rename: never a partial file under the final name */
    char tmp[1500]; snprintf(tmp, sizeof tmp, "%s.part", path);
    FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    int ok = fwrite(buf, 1, n, f) == n && !fflush(f);
    /* written, then out of the page cache: a bulk copy must not evict the cached data of jobs running beside it */
    if (ok && n > 65536) { const int fd = fileno(f); ok = !fdatasync(fd); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); }
    if (fclose(f) || !ok) { unlink(tmp); return -1; }
    return rename(tmp, path) ? -1 : 0;
}
static int64_t file_size(const char *p) { struct stat st; return stat(p, &st) ? -1 : (int64_t)st.st_size; }

/* ---------------------------------------------------------------- mirror ---------------------------------------------------------------- */

typedef struct {
    store *s; const char *key, *out, *link; const z3_meta *m; int64_t g[3]; int64_t total;
    double rate, t_avail, t_file; const char *rate_file; pthread_mutex_t mu;   /* bytes / s (0: unlimited), the token bucket's next free time */
    atomic_long next, have, linked, got, missing, failed, bytes; double t0; atomic_long last_print;
    uint8_t *present;   /* per shard: 1 present after the copy */
} mir_t;

static void shard_rel(const z3_meta *m, int64_t k, const int64_t g[3], char *buf, size_t n) {
    const int64_t z = k / (g[1] * g[2]), y = (k / g[2]) % g[1], x = k % g[2]; const char sep = m->sep ? m->sep : '/';
    snprintf(buf, n, "c%c%lld%c%lld%c%lld", sep, (long long)z, sep, (long long)y, sep, (long long)x);
}
static void mir_throttle(mir_t *M, int64_t bytes) {
    pthread_mutex_lock(&M->mu);
    const double t = snow();
    if (M->rate_file && t - M->t_file >= 10) {   /* --mbps-file: re-read the cap every 10 s (a watchdog lowers it while trainers slow down) */
        M->t_file = t; FILE *f = fopen(M->rate_file, "r"); double v;
        if (f) { if (fscanf(f, "%lf", &v) == 1 && v >= 0 && v * 1e6 != M->rate) { fprintf(stderr, "mirror: rate cap %.0f MB/s (%s)\n", v, M->rate_file); M->rate = v * 1e6; } fclose(f); }
    }
    if (M->rate <= 0) { pthread_mutex_unlock(&M->mu); return; }
    if (M->t_avail < t) M->t_avail = t;
    M->t_avail += (double)bytes / M->rate; const double wait = M->t_avail - t - 1.0;   /* (one second of burst) */
    pthread_mutex_unlock(&M->mu);
    if (wait > 0) { struct timespec ts = {(time_t)wait, (long)((wait - floor(wait)) * 1e9)}; nanosleep(&ts, nullptr); }
}
static void *mir_worker(void *arg) {
    mir_t *M = arg; char rel[256], rk[1400], lp[1400], tp[1400];
    for (int64_t k; (k = atomic_fetch_add(&M->next, 1)) < M->total;) {
        shard_rel(M->m, k, M->g, rel, sizeof rel);
        snprintf(rk, sizeof rk, "%s/%s", M->key, rel); snprintf(lp, sizeof lp, "%s/%s", M->out, rel);
        int64_t rs = -1;
        for (int a = 0; a < 4 && rs == -1; a++) { rs = store_size(M->s, rk); if (rs == -1) sleep(1u << a); }
        if (rs == -2) { atomic_fetch_add(&M->missing, 1); continue; }
        if (rs < 0) { fprintf(stderr, "mirror: %s: size failed (HTTP %ld)\n", rk, store_last_status()); atomic_fetch_add(&M->failed, 1); continue; }
        if (file_size(lp) == rs) { atomic_fetch_add(&M->have, 1); M->present[k] = 1; continue; }
        char dir[1400]; snprintf(dir, sizeof dir, "%s", lp); char *sl = strrchr(dir, '/'); if (sl) { *sl = 0; smkdirs(dir); }
        if (M->link) {
            snprintf(tp, sizeof tp, "%s/%s", M->link, rel);
            if (file_size(tp) == rs) { unlink(lp); if (!link(tp, lp)) { atomic_fetch_add(&M->linked, 1); M->present[k] = 1; continue; } }
        }
        uint8_t *buf = malloc((size_t)(rs ? rs : 1)); int ok = 0;
        if (buf) for (int a = 0; a < 4 && !ok; a++) {
            mir_throttle(M, rs);
            ok = rs == 0 || store_read(M->s, rk, 0, rs, buf) == rs;
            if (!ok) sleep(1u << a);
        }
        if (!ok || write_file(lp, buf, (size_t)rs)) { fprintf(stderr, "mirror: %s: download / write failed\n", rk); atomic_fetch_add(&M->failed, 1); free(buf); continue; }
        free(buf);
        atomic_fetch_add(&M->got, 1); atomic_fetch_add(&M->bytes, rs); M->present[k] = 1;
        const long t = (long)(snow() - M->t0), lp0 = atomic_load(&M->last_print);
        if (t - lp0 >= 30 && atomic_compare_exchange_strong(&M->last_print, &(long){lp0}, t))
            fprintf(stderr, "mirror: %lld / %lld shards visited, %ld downloaded (%.2f GB, %.1f MB/s), %ld linked, %ld present, %ld missing\n",
                    (long long)atomic_load(&M->next), (long long)M->total, atomic_load(&M->got), atomic_load(&M->bytes) / 1e9,
                    atomic_load(&M->bytes) / 1e6 / (snow() - M->t0), atomic_load(&M->linked), atomic_load(&M->have), atomic_load(&M->missing));
    }
    return nullptr;
}
static uint64_t rng_next(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }

int cmd_mirror(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm mirror <root> <array-key> <out-dir> [--threads 4] [--mbps 0] [--mbps-file F] [--link-from DIR] [--verify N] [--seed 1]\n"); return 2; }
    mir_t M = {0};
    M.key = argv[3]; M.out = argv[4]; M.link = sopt(argc, argv, "--link-from", nullptr);
    const int nt = atoi(sopt(argc, argv, "--threads", "4")), nver = atoi(sopt(argc, argv, "--verify", "0"));
    M.rate = atof(sopt(argc, argv, "--mbps", "0")) * 1e6; M.rate_file = sopt(argc, argv, "--mbps-file", nullptr); M.t_file = -1e9; uint64_t seed = (uint64_t)atoll(sopt(argc, argv, "--seed", "1"));
    store_global_init();
    M.s = store_open(argv[2]);
    z3 *z = M.s ? z3_open(M.s, M.key, nullptr) : nullptr;
    if (!z) { fprintf(stderr, "mirror: cannot open %s/%s: %s\n", argv[2], M.key, z3_error()); return 1; }
    M.m = z3_meta_of(z);
    if (!M.m->sharded) { fprintf(stderr, "mirror: %s is not sharded (per-chunk files): not supported\n", M.key); return 1; }
    if (smkdirs(M.out)) { fprintf(stderr, "mirror: cannot create %s\n", M.out); return 1; }
    {   /* the array metadata, byte for byte */
        char k[1400], p[1400]; snprintf(k, sizeof k, "%s/zarr.json", M.key); snprintf(p, sizeof p, "%s/zarr.json", M.out);
        size_t n = 0; uint8_t *b = store_read_all(M.s, k, &n);
        if (!b || write_file(p, b, n)) { fprintf(stderr, "mirror: cannot copy %s\n", k); free(b); return 1; }
        free(b);
    }
    for (int d = 0; d < 3; d++) M.g[d] = (M.m->shape[d] + M.m->shard[d] - 1) / M.m->shard[d];
    M.total = M.g[0] * M.g[1] * M.g[2]; M.present = calloc((size_t)M.total, 1); M.t0 = snow();
    pthread_mutex_init(&M.mu, nullptr);
    const int T = nt < 1 ? 1 : nt > 64 ? 64 : nt; pthread_t th[64];
    for (int i = 0; i < T; i++) pthread_create(&th[i], nullptr, mir_worker, &M);
    for (int i = 0; i < T; i++) pthread_join(th[i], nullptr);
    const long failed = atomic_load(&M.failed);
    fprintf(stderr, "mirror: %s: %lld shards: %ld already present, %ld linked, %ld downloaded (%.3f GB in %.0f s), %ld absent (fill), %ld failed\n",
            M.key, (long long)M.total, atomic_load(&M.have), atomic_load(&M.linked), atomic_load(&M.got), atomic_load(&M.bytes) / 1e9, snow() - M.t0,
            atomic_load(&M.missing), failed);
    int bad = failed > 0;
    if (nver > 0 && !bad) {   /* random present inner chunks: decoded local == decoded remote; the shard files' crc32c */
        int64_t np = 0; for (int64_t k = 0; k < M.total; k++) np += M.present[k];
        char par[1400], base[400]; snprintf(par, sizeof par, "%s", M.out); char *sl = strrchr(par, '/');
        if (sl) { snprintf(base, sizeof base, "%s", sl + 1); *sl = 0; if (!*par) snprintf(par, sizeof par, "/"); } else { snprintf(base, sizeof base, "%s", par); snprintf(par, sizeof par, "."); }
        store *ls = store_open(par); z3 *lz = ls ? z3_open(ls, base, nullptr) : nullptr;
        if (!lz) { fprintf(stderr, "mirror: verify: cannot open the local copy %s: %s\n", M.out, z3_error()); return 1; }
        const int C = M.m->chunk[0]; uint8_t *a = malloc((size_t)C * C * C), *b = malloc((size_t)C * C * C);
        int nok = 0, nfile = 0;
        for (int v = 0; v < nver && np > 0; v++) {
            int64_t k = 0, o[3], n[3] = {C, C, C}; int rerr = 0;
            for (int tries = 0; tries < 32; tries++) {   /* a chunk with data (up to 32 draws; all-fill chunks prove little) */
                int64_t pick = (int64_t)(rng_next(&seed) % (uint64_t)np); k = 0;
                for (; k < M.total; k++) if (M.present[k] && pick-- == 0) break;
                const int64_t sc[3] = {k / (M.g[1] * M.g[2]), (k / M.g[2]) % M.g[1], k % M.g[2]};
                for (int d = 0; d < 3; d++) {
                    const int64_t lo = sc[d] * M.m->shard[d], hi = lo + M.m->shard[d] < M.m->shape[d] ? lo + M.m->shard[d] : M.m->shape[d];
                    const int64_t nc = (hi - lo + C - 1) / C; o[d] = lo + (int64_t)(rng_next(&seed) % (uint64_t)nc) * C;
                }
                if ((rerr = z3_read(lz, o, n, a, 1))) break;
                size_t nz0 = 0; for (size_t q = 0; q < (size_t)C * C * C && !nz0; q++) nz0 = a[q] != 0;
                if (nz0) break;
            }
            if (rerr || z3_read(z, o, n, b, 1)) { fprintf(stderr, "mirror: verify: read failed at %lld,%lld,%lld: %s\n", (long long)o[0], (long long)o[1], (long long)o[2], z3_error()); bad = 1; continue; }
            const uint32_t ca = crc32c(0, a, (size_t)C * C * C), cb = crc32c(0, b, (size_t)C * C * C); const int same = !memcmp(a, b, (size_t)C * C * C);
            size_t nz = 0; for (size_t q = 0; q < (size_t)C * C * C; q++) nz += a[q] != 0;
            fprintf(stderr, "mirror: verify chunk %lld,%lld,%lld: local crc %08x remote %08x %s (%.1f%% nonzero)\n", (long long)o[0], (long long)o[1], (long long)o[2],
                    ca, cb, same ? "same" : "DIFFERENT", 100.0 * nz / ((double)C * C * C));
            nok += same; bad |= !same;
            if (nfile < 4) {   /* the whole shard file: local bytes vs a fresh download */
                char rel[256], rk[1400], lp[1400]; shard_rel(M.m, k, M.g, rel, sizeof rel);
                snprintf(rk, sizeof rk, "%s/%s", M.key, rel); snprintf(lp, sizeof lp, "%s/%s", M.out, rel);
                size_t rn = 0; uint8_t *rb = store_read_all(M.s, rk, &rn); FILE *f = fopen(lp, "rb"); uint8_t *lb = rb ? malloc(rn + 1) : nullptr;
                const int got = f && lb && fread(lb, 1, rn + 1, f) == rn; if (f) fclose(f);
                const uint32_t cr = rb ? crc32c(0, rb, rn) : 0, cl = got ? crc32c(0, lb, rn) : 1;
                fprintf(stderr, "mirror: verify shard %s: %zu bytes, local crc %08x remote %08x %s\n", rel, rn, cl, cr, got && cl == cr ? "same" : "DIFFERENT");
                bad |= !(got && cl == cr); free(rb); free(lb); nfile++;
            }
        }
        fprintf(stderr, "mirror: verify: %d / %d chunks identical, %d shard files compared: %s\n", nok, nver, nfile, bad ? "FAILED" : "ok");
        free(a); free(b); z3_close(lz); store_close(ls);
    }
    free(M.present); z3_close(z);
    return bad ? 1 : 0;
}

/* ---------------------------------------------------------------- scanprep ---------------------------------------------------------------- */

int sp_otsu(const double *hist, int nb) {
    double tot = 0, sum = 0; int pop = 0;
    for (int i = 0; i < nb; i++) { tot += hist[i]; sum += (double)i * hist[i]; pop += hist[i] > 0; }
    if (pop < 2) return -1;
    double w0 = 0, s0 = 0, best = -1; int bt = -1, bt2 = -1;
    for (int t = 1; t < nb; t++) {
        w0 += hist[t - 1]; s0 += (double)(t - 1) * hist[t - 1];
        const double w1 = tot - w0; if (w0 <= 0 || w1 <= 0) continue;
        const double m0 = s0 / w0, m1 = (sum - s0) / w1, v = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (v > best * (1 + 1e-12)) { best = v; bt = t; bt2 = t; } else if (v >= best * (1 - 1e-12) && bt2 == t - 1) bt2 = t;   /* a plateau (empty bins between the modes): its middle */
    }
    return bt < 0 ? -1 : (bt + bt2) / 2;
}

int sp_close(uint8_t *m, const int64_t n[3], float r, int nthreads) {
    const size_t N = (size_t)n[0] * n[1] * n[2];
    if (r <= 0) return 0;
    float *d = malloc(N * sizeof *d); uint8_t *f = calloc(N, 1);
    if (!d || !f) { free(d); free(f); return -1; }
    const float r2 = r * r;
    for (size_t k = 0; k < N; k++) f[k] = m[k] != 0;
    if (edt_sq(f, n, d, nthreads)) { free(d); free(f); return -1; }
    for (size_t k = 0; k < N; k++) f[k] = !(d[k] <= r2);   /* background of the dilation */
    if (edt_sq(f, n, d, nthreads)) { free(d); free(f); return -1; }
    for (size_t k = 0; k < N; k++) m[k] = !f[k] && d[k] > r2;   /* erode: dilated cells farther than r from its background */
    free(d); free(f);
    return 0;
}

int sp_fill_holes(uint8_t *m, const int64_t n[3], int plane2d) {
    const int64_t nz = n[0], ny = n[1], nx = n[2]; const size_t N = (size_t)nz * ny * nx;
    uint32_t *q = N < 4294967295u ? malloc(N * sizeof *q) : nullptr;
    if (!q) return -1;
    /* 3-D: background reachable from the border through background is marked 2 */
    size_t qh = 0, qt = 0;
#define PUSH(k) do { if (!m[k]) { m[k] = 2; q[qt++] = (uint32_t)(k); } } while (0)
    for (int64_t z = 0; z < nz; z++) for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++)
        if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || x == 0 || x == nx - 1) PUSH(((size_t)z * ny + y) * nx + x);
    while (qh < qt) {
        const size_t k = q[qh++]; const int64_t z = (int64_t)(k / ((size_t)ny * nx)), y = (int64_t)((k / nx) % ny), x = (int64_t)(k % nx);
        if (z > 0) { PUSH(k - (size_t)ny * nx); } if (z < nz - 1) { PUSH(k + (size_t)ny * nx); }
        if (y > 0) { PUSH(k - nx); } if (y < ny - 1) { PUSH(k + nx); }
        if (x > 0) { PUSH(k - 1); } if (x < nx - 1) { PUSH(k + 1); }
    }
    for (size_t k = 0; k < N; k++) m[k] = m[k] != 2;   /* unreached background -> foreground */
    if (plane2d) for (int64_t z = 0; z < nz; z++) {
        uint8_t *p = m + (size_t)z * ny * nx; qh = qt = 0;
#define PUSH2(k) do { if (!p[k]) { p[k] = 2; q[qt++] = (uint32_t)(k); } } while (0)
        for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++) if (y == 0 || y == ny - 1 || x == 0 || x == nx - 1) PUSH2((size_t)y * nx + x);
        while (qh < qt) {
            const size_t k = q[qh++]; const int64_t y = (int64_t)(k / nx), x = (int64_t)(k % nx);
            if (y > 0) { PUSH2(k - nx); } if (y < ny - 1) { PUSH2(k + nx); }
            if (x > 0) { PUSH2(k - 1); } if (x < nx - 1) { PUSH2(k + 1); }
        }
        for (size_t k = 0; k < (size_t)ny * nx; k++) p[k] = p[k] != 2;
#undef PUSH2
    }
#undef PUSH
    free(q);
    return 0;
}

/* BFS of the 6-connected foreground component of seed s whose cells hold value from, setting them to value to; returns its size */
static size_t sp_flood(uint8_t *m, const int64_t n[3], size_t s, uint8_t from, uint8_t to, uint32_t *q) {
    const int64_t ny = n[1], nx = n[2], nz = n[0]; size_t qh = 0, qt = 0;
    m[s] = to; q[qt++] = (uint32_t)s;
#define PUSHC(k) do { const size_t k_ = (k); if (m[k_] == from) { m[k_] = to; q[qt++] = (uint32_t)k_; } } while (0)
    while (qh < qt) {
        const size_t k = q[qh++]; const int64_t z = (int64_t)(k / ((size_t)ny * nx)), y = (int64_t)((k / nx) % ny), x = (int64_t)(k % nx);
        if (z > 0) { PUSHC(k - (size_t)ny * nx); } if (z < nz - 1) { PUSHC(k + (size_t)ny * nx); }
        if (y > 0) { PUSHC(k - nx); } if (y < ny - 1) { PUSHC(k + nx); }
        if (x > 0) { PUSHC(k - 1); } if (x < nx - 1) { PUSHC(k + 1); }
    }
#undef PUSHC
    return qt;
}

int64_t sp_keep_components(uint8_t *m, const int64_t n[3], double frac, int64_t *ncomp) {
    const size_t N = (size_t)n[0] * n[1] * n[2];
    uint32_t *q = N < 4294967295u ? malloc(N * sizeof *q) : nullptr;
    if (!q) return -1;
    size_t cap = 256, nc = 0, *seed = malloc(cap * sizeof *seed), *size = malloc(cap * sizeof *size), big = 0;
    for (size_t k = 0; k < N; k++) {
        if (m[k] != 1) continue;
        const size_t s = sp_flood(m, n, k, 1, 2, q);
        if (nc == cap) { cap *= 2; seed = realloc(seed, cap * sizeof *seed); size = realloc(size, cap * sizeof *size); }
        seed[nc] = k; size[nc++] = s; if (s > big) big = s;
    }
    int64_t kept = 0;
    for (size_t c = 0; c < nc; c++) {
        if ((double)size[c] >= frac * (double)big) { kept++; continue; }
        sp_flood(m, n, seed[c], 2, 0, q);
    }
    for (size_t k = 0; k < N; k++) m[k] = m[k] == 2;
    if (ncomp) *ncomp = (int64_t)nc;
    free(q); free(seed); free(size);
    return kept;
}

int64_t sp_valid_centres(const uint8_t *m, const int64_t n[3], int w, double min_inside, uint8_t *out) {
    const int64_t nz = n[0], ny = n[1], nx = n[2]; const int64_t sy = nx + 1, sz = (ny + 1) * (nx + 1);
    uint32_t *S = calloc((size_t)(nz + 1) * (size_t)sz, sizeof *S);   /* 3-D prefix sums (exclusive) */
    if (!S) return -1;
    for (int64_t z = 0; z < nz; z++) for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++) {
        const int64_t i = (z + 1) * sz + (y + 1) * sy + (x + 1);
        S[i] = (uint32_t)(m[((size_t)z * ny + y) * nx + x] != 0) + S[i - 1] + S[i - sy] + S[i - sz] - S[i - sy - 1] - S[i - sz - 1] - S[i - sz - sy] + S[i - sz - sy - 1];
    }
    if (w < 1) w = 1;
    const double need = min_inside * (double)w * w * w; int64_t nv = 0;
    for (int64_t z = 0; z < nz; z++) for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++) {
        const size_t k = ((size_t)z * ny + y) * nx + x;
        if (!m[k]) { out[k] = 0; continue; }
        int64_t lo[3] = {z - w / 2, y - w / 2, x - w / 2}, hi[3], nn[3] = {nz, ny, nx};
        for (int d = 0; d < 3; d++) { hi[d] = lo[d] + w; if (lo[d] < 0) lo[d] = 0; if (hi[d] > nn[d]) hi[d] = nn[d]; }
        const double c = (double)S[hi[0] * sz + hi[1] * sy + hi[2]] - S[lo[0] * sz + hi[1] * sy + hi[2]] - S[hi[0] * sz + lo[1] * sy + hi[2]] - S[hi[0] * sz + hi[1] * sy + lo[2]]
                       + S[lo[0] * sz + lo[1] * sy + hi[2]] + S[lo[0] * sz + hi[1] * sy + lo[2]] + S[hi[0] * sz + lo[1] * sy + lo[2]] - S[lo[0] * sz + lo[1] * sy + lo[2]];
        out[k] = c >= need ? 2 : 1; nv += out[k] == 2;
    }
    free(S);
    return nv;
}

double sp_hist_quantile(const uint64_t *h, double q) {
    double tot = 0; for (int i = 0; i < 256; i++) tot += (double)h[i];
    if (tot <= 0) return -1;
    const double want = q * tot; double c = 0;
    for (int i = 0; i < 256; i++) { c += (double)h[i]; if (c >= want && h[i]) return i; }
    return 255;
}

/* the voxel statistics: random inner chunks of level L, nonzero voxels of object cells */
typedef struct { z3 *z; const uint8_t *mask; int64_t G[3]; int cell, C; const int64_t *chunks; int64_t nch; atomic_long next; uint64_t hist[64][256]; int failed; } st_t;
typedef struct { st_t *S; int id; } st_arg;
static void *st_worker(void *p) {
    st_arg *a = p; st_t *S = a->S; const int C = S->C; const z3_meta *m = z3_meta_of(S->z);
    uint8_t *b = malloc((size_t)C * C * C); uint64_t *h = S->hist[a->id];
    if (!b) { S->failed = 1; return nullptr; }
    for (int64_t i; (i = atomic_fetch_add(&S->next, 1)) < S->nch;) {
        const int64_t ci = S->chunks[i]; int64_t gc[3];
        for (int d = 0; d < 3; d++) gc[d] = (m->shape[d] + C - 1) / C;
        const int64_t o[3] = {ci / (gc[1] * gc[2]) * C, (ci / gc[2]) % gc[1] * C, ci % gc[2] * C}; int64_t n[3];
        for (int d = 0; d < 3; d++) n[d] = o[d] + C <= m->shape[d] ? C : m->shape[d] - o[d];
        if (z3_read(S->z, o, n, b, 1)) { S->failed = 1; break; }
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) {
            const uint8_t *row = b + ((size_t)z * n[1] + y) * n[2];
            const size_t mrow = ((size_t)((o[0] + z) / S->cell) * S->G[1] + (size_t)((o[1] + y) / S->cell)) * S->G[2];
            for (int64_t x = 0; x < n[2]; x++) if (row[x] && S->mask[mrow + (size_t)((o[2] + x) / S->cell)]) h[row[x]]++;
        }
    }
    free(b);
    return nullptr;
}

static double um_from_name(const char *key) {   /* "...-8.640um-..." */
    const char *p = strstr(key, "um"), *best = nullptr;
    for (; p; p = strstr(p + 1, "um")) { const char *q = p; while (q > key && (q[-1] == '.' || (q[-1] >= '0' && q[-1] <= '9'))) q--; if (q < p) best = q; }
    return best ? atof(best) : 0;
}

int cmd_scanprep(int argc, char **argv) {
    if (argc < 4 || !sopt(argc, argv, "--out", nullptr) || !sopt(argc, argv, "--level", nullptr)) {
        fprintf(stderr, "usage: ufsm scanprep <root> <group-key> --level L --out PREFIX [--um U] [--cell 16] [--close-mm 2] [--window 128] [--min-inside 0.25]\n"
                        "                     [--fill2d 1] [--stats-chunks 256] [--threads 16] [--seed 1]\n");
        return 2;
    }
    const char *root = argv[2], *group = argv[3], *out = sopt(argc, argv, "--out", nullptr);
    const int L = atoi(sopt(argc, argv, "--level", "0")), cell = atoi(sopt(argc, argv, "--cell", "16")), win = atoi(sopt(argc, argv, "--window", "128"));
    const int fill2d = atoi(sopt(argc, argv, "--fill2d", "1")), nstat = atoi(sopt(argc, argv, "--stats-chunks", "256")), nt = atoi(sopt(argc, argv, "--threads", "16"));
    const double close_mm = atof(sopt(argc, argv, "--close-mm", "2")), min_inside = atof(sopt(argc, argv, "--min-inside", "0.25"));
    const double keep_frac = atof(sopt(argc, argv, "--keep-frac", "0.1"));
    double zr[2] = {-1e18, 1e18};
    if (sopt(argc, argv, "--zrange", nullptr) && sscanf(sopt(argc, argv, "--zrange", ""), "%lf,%lf", zr, zr + 1) != 2) { fprintf(stderr, "scanprep: --zrange z0,z1 (level-L voxels)\n"); return 2; }
    uint64_t seed = (uint64_t)atoll(sopt(argc, argv, "--seed", "1"));
    double um0 = atof(sopt(argc, argv, "--um", "0")); if (um0 <= 0) um0 = um_from_name(group);
    if (um0 <= 0 || cell < 1 || (cell & (cell - 1)) || cell > 128 || L < 0 || L >= MAXLEV) { fprintf(stderr, "scanprep: need --um (not in the name), --cell a power of 2 <= 128, 0 <= --level < %d\n", MAXLEV); return 2; }
    const double t0 = snow();
    store_global_init();
    store *st = store_open(root);
    z3 *zl = st ? pyramid_open_level(st, group, L, um0, nullptr) : nullptr;
    if (!zl) { fprintf(stderr, "scanprep: cannot open level %d of %s/%s: %s\n", L, root, group, z3_error()); return 1; }
    const z3_meta *ml = z3_meta_of(zl);
    int lc = L, f = cell; z3 *zc = zl;   /* the coarse level the cell means come from: L+2, L+3, L+4 or L */
    for (int k = 2; k <= 4; k++) { if ((1 << k) > cell || L + k >= MAXLEV) break; z3 *t = pyramid_open_level(st, group, L + k, um0, nullptr); if (t) { zc = t; lc = L + k; f = cell >> k; break; } }
    const z3_meta *mc = z3_meta_of(zc);
    int64_t G[3]; for (int d = 0; d < 3; d++) G[d] = (ml->shape[d] + cell - 1) / cell;
    const size_t NG = (size_t)G[0] * G[1] * G[2], plane = (size_t)G[1] * G[2];
    if (NG >= 4294967295u) { fprintf(stderr, "scanprep: %zu cells: too many (raise --cell)\n", NG); return 1; }
    fprintf(stderr, "scanprep: %s level %d (%.3f um) shape %lldx%lldx%lld, cells of %d voxels: grid %lldx%lldx%lld, cell means from level %d (pool %d)\n", group, L, um0 * (1 << L),
            (long long)ml->shape[0], (long long)ml->shape[1], (long long)ml->shape[2], cell, (long long)G[0], (long long)G[1], (long long)G[2], lc, f);
    /* cell means over nonzero voxels: cm 0 = no nonzero voxel, else max(1, round(mean)) */
    uint8_t *cm = calloc(NG, 1), *slab = malloc((size_t)f * mc->shape[1] * mc->shape[2]); double *ss = malloc(plane * sizeof *ss); uint32_t *cc = malloc(plane * sizeof *cc);
    if (!cm || !slab || !ss || !cc) { fprintf(stderr, "scanprep: out of memory\n"); return 1; }
    for (int64_t gz = 0; gz < G[0]; gz++) {
        const int64_t z0 = gz * f; if (z0 >= mc->shape[0]) break;
        const int64_t o[3] = {z0, 0, 0}, n[3] = {z0 + f <= mc->shape[0] ? f : mc->shape[0] - z0, mc->shape[1], mc->shape[2]};
        if (z3_read(zc, o, n, slab, nt)) { fprintf(stderr, "scanprep: read failed: %s\n", z3_error()); return 1; }
        memset(ss, 0, plane * sizeof *ss); memset(cc, 0, plane * sizeof *cc);
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) {
            const int64_t gy = y / f; if (gy >= G[1]) continue;
            const uint8_t *row = slab + ((size_t)z * n[1] + y) * n[2]; double *sr = ss + (size_t)gy * G[2]; uint32_t *cr = cc + (size_t)gy * G[2];
            for (int64_t x = 0; x < n[2]; x++) if (row[x]) { const int64_t gx = x / f; if (gx < G[2]) { sr[gx] += row[x]; cr[gx]++; } }
        }
        uint8_t *cp = cm + (size_t)gz * plane;
        for (size_t k = 0; k < plane; k++) if (cc[k]) { const long v = lround(ss[k] / cc[k]); cp[k] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v); }
    }
    free(slab); free(ss); free(cc);
    double ch[256] = {0}; int64_t nzc = 0;
    for (size_t k = 0; k < NG; k++) if (cm[k]) { ch[cm[k]]++; nzc++; }
    const int cut = sp_otsu(ch, 256);
    if (cut < 0) { fprintf(stderr, "scanprep: no nonzero cells (or one value): nothing to mask\n"); return 1; }
    uint8_t *mk = malloc(NG), *val = malloc(NG);
    if (!mk || !val) { fprintf(stderr, "scanprep: out of memory\n"); return 1; }
    int64_t n_otsu = 0; for (size_t k = 0; k < NG; k++) { mk[k] = cm[k] >= cut; n_otsu += mk[k]; }
    const float rc = (float)(close_mm * 1000.0 / (um0 * (1 << L) * cell));
    int64_t ncomp = 0, nkept = 0;
    if (sp_close(mk, G, rc, nt) || (keep_frac > 0 && (nkept = sp_keep_components(mk, G, keep_frac, &ncomp)) < 0) || sp_fill_holes(mk, G, fill2d)) { fprintf(stderr, "scanprep: out of memory (closing / components / hole fill)\n"); return 1; }
    for (int64_t gz = 0; gz < G[0]; gz++) {   /* --zrange: cells whose centre lies outside [z0, z1) (level-L voxels) are dropped (sample mounts at the z ends) */
        const double zc = (gz + 0.5) * cell; if (zc >= zr[0] && zc < zr[1]) continue;
        memset(mk + (size_t)gz * plane, 0, plane);
    }
    int64_t nobj = 0; for (size_t k = 0; k < NG; k++) { mk[k] = mk[k] && cm[k]; nobj += mk[k]; }
    const int wc = (win + cell - 1) / cell;
    const int64_t nvalid = sp_valid_centres(mk, G, wc, min_inside, val);
    if (nvalid < 0) { fprintf(stderr, "scanprep: out of memory (valid centres)\n"); return 1; }
    fprintf(stderr, "scanprep: %lld nonzero cells, Otsu cut %d -> %lld, closing r %.1f cells, %lld of %lld components kept (>= %.2f x the largest) + hole fill -> %lld object cells, %lld valid window centres (%d^3 cells, min inside %.2f)\n",
            (long long)nzc, cut, (long long)n_otsu, rc, (long long)nkept, (long long)ncomp, keep_frac, (long long)nobj, (long long)nvalid, wc, min_inside);
    /* voxel statistics from random inner chunks holding object cells */
    st_t *S = calloc(1, sizeof *S);
    S->z = zl; S->mask = mk; memcpy(S->G, G, sizeof G); S->cell = cell; S->C = ml->chunk[0] > 0 ? ml->chunk[0] : 128;
    int64_t gc[3]; for (int d = 0; d < 3; d++) gc[d] = (ml->shape[d] + S->C - 1) / S->C;
    int64_t *chs = nullptr, nch = 0;
    if (nobj > 0 && nstat > 0) {
        const int64_t want = (int64_t)nstat * 8; chs = malloc((size_t)want * sizeof *chs);
        for (int64_t i = 0; i < want; i++) {   /* random object cells (rejection over the grid) -> their chunks */
            size_t k = 0; for (int a = 0; a < 1000; a++) { k = (size_t)(rng_next(&seed) % NG); if (mk[k]) break; }
            if (!mk[k]) continue;
            const int64_t c[3] = {(int64_t)(k / plane) * cell / S->C, (int64_t)((k / G[2]) % G[1]) * cell / S->C, (int64_t)(k % G[2]) * cell / S->C};
            chs[nch++] = (c[0] * gc[1] + c[1]) * gc[2] + c[2];
        }
        for (int64_t i = 1; i < nch; i++) { const int64_t v = chs[i]; int64_t j = i - 1; while (j >= 0 && chs[j] > v) { chs[j + 1] = chs[j]; j--; } chs[j + 1] = v; }
        int64_t u = 0; for (int64_t i = 0; i < nch; i++) if (!u || chs[u - 1] != chs[i]) chs[u++] = chs[i];
        nch = u;
        for (int64_t i = nch - 1; i > 0; i--) { const int64_t j = (int64_t)(rng_next(&seed) % (uint64_t)(i + 1)); const int64_t t = chs[i]; chs[i] = chs[j]; chs[j] = t; }
        if (nch > nstat) nch = nstat;
    }
    S->chunks = chs; S->nch = nch;
    const int T = nt < 1 ? 1 : nt > 64 ? 64 : nt; pthread_t th[64]; st_arg sa[64];
    for (int i = 0; i < T; i++) { sa[i] = (st_arg){S, i}; pthread_create(&th[i], nullptr, st_worker, &sa[i]); }
    for (int i = 0; i < T; i++) pthread_join(th[i], nullptr);
    if (S->failed) { fprintf(stderr, "scanprep: statistics read failed: %s\n", z3_error()); return 1; }
    uint64_t h[256] = {0}; for (int i = 0; i < T; i++) for (int v = 0; v < 256; v++) h[v] += S->hist[i][v];
    double n = 0, sum = 0, sq = 0; for (int v = 0; v < 256; v++) { n += (double)h[v]; sum += (double)v * h[v]; sq += (double)v * v * h[v]; }
    const double mean = n ? sum / n : 0, sd = n ? sqrt(fmax(sq / n - mean * mean, 0)) : 0, med = sp_hist_quantile(h, 0.5);
    uint64_t hd[256] = {0}; for (int v = 0; v < 256; v++) hd[abs(v - (int)med)] += h[v];
    const double mad = sp_hist_quantile(hd, 0.5), scale = mad > 0 ? 1.4826 * mad : sd;
    double hf[256]; for (int v = 0; v < 256; v++) hf[v] = (double)h[v];
    const int vcut = sp_otsu(hf, 256);
    fprintf(stderr, "scanprep: %lld chunks, %.0f nonzero object voxels: p0.5 %.0f, median %.0f, p99.5 %.0f, MAD %.0f (scale %.2f), mean %.2f, sd %.2f, Otsu %d (%.1f s)\n",
            (long long)nch, n, sp_hist_quantile(h, 0.005), med, sp_hist_quantile(h, 0.995), mad, scale, mean, sd, vcut, snow() - t0);
    /* outputs */
    char p[1400];
    snprintf(p, sizeof p, "%s.mask.raw", out);
    if (write_file(p, val, NG)) { fprintf(stderr, "scanprep: cannot write %s\n", p); return 1; }
    int lev = L; for (int c = cell; c > 1; c >>= 1) lev++;
    snprintf(p, sizeof p, "%s.mask.raw.json", out);
    FILE *fj = fopen(p, "w"); if (!fj) { fprintf(stderr, "scanprep: cannot write %s\n", p); return 1; }
    fprintf(fj, "{\"grid\": [%lld, %lld, %lld], \"cell\": %d, \"level\": %d, \"lev\": %d, \"codes\": \"0 outside, 1 object, 2 valid window centre\"}\n",
            (long long)G[0], (long long)G[1], (long long)G[2], cell, L, lev);
    fclose(fj);
    {   /* preview: mid z plane (G1 x G2) above the mid y plane (G0 x G2); cell means left, mask codes right */
        const int64_t W = 2 * G[2], H = G[1] + G[0]; uint8_t *img = calloc((size_t)(W * H), 1);
        for (int64_t y = 0; y < G[1]; y++) for (int64_t x = 0; x < G[2]; x++) { const size_t k = (size_t)(G[0] / 2) * plane + (size_t)y * G[2] + x;
            img[y * W + x] = cm[k]; img[y * W + G[2] + x] = (uint8_t)(val[k] * 127); }
        for (int64_t z = 0; z < G[0]; z++) for (int64_t x = 0; x < G[2]; x++) { const size_t k = (size_t)z * plane + (size_t)(G[1] / 2) * G[2] + x;
            img[(G[1] + z) * W + x] = cm[k]; img[(G[1] + z) * W + G[2] + x] = (uint8_t)(val[k] * 127); }
        snprintf(p, sizeof p, "%s.mask.pgm", out); FILE *fp = fopen(p, "wb");
        if (fp) { fprintf(fp, "P5\n%lld %lld\n255\n", (long long)W, (long long)H); fwrite(img, 1, (size_t)(W * H), fp); fclose(fp); }
        free(img);
    }
    snprintf(p, sizeof p, "%s.json", out);
    fj = fopen(p, "w"); if (!fj) { fprintf(stderr, "scanprep: cannot write %s\n", p); return 1; }
    fprintf(fj, "{\n  \"root\": \"%s\", \"group\": \"%s\", \"um\": %.4f, \"level\": %d, \"level_um\": %.4f,\n", root, group, um0, L, um0 * (1 << L));
    fprintf(fj, "  \"shape\": [%lld, %lld, %lld], \"cell\": %d, \"grid\": [%lld, %lld, %lld], \"lev\": %d, \"mean_level\": %d,\n",
            (long long)ml->shape[0], (long long)ml->shape[1], (long long)ml->shape[2], cell, (long long)G[0], (long long)G[1], (long long)G[2], lev, lc);
    fprintf(fj, "  \"cell_cut\": %d, \"close_mm\": %.3f, \"close_cells\": %.2f, \"keep_frac\": %.3f, \"components\": %lld, \"kept\": %lld, \"fill2d\": %d, \"window\": %d, \"min_inside\": %.3f,\n", cut, close_mm, rc, keep_frac, (long long)ncomp, (long long)nkept, fill2d, win, min_inside);
    fprintf(fj, "  \"nonzero_cells\": %lld, \"otsu_cells\": %lld, \"object_cells\": %lld, \"valid_cells\": %lld,\n", (long long)nzc, (long long)n_otsu, (long long)nobj, (long long)nvalid);
    fprintf(fj, "  \"mask\": \"%s.mask.raw\",\n", out);
    fprintf(fj, "  \"stats\": {\"chunks\": %lld, \"voxels\": %.0f, \"p0_5\": %.0f, \"p25\": %.0f, \"median\": %.0f, \"p75\": %.0f, \"p99_5\": %.0f, \"mad\": %.0f, \"scale\": %.4f, \"mean\": %.4f, \"sd\": %.4f, \"otsu\": %d},\n",
            (long long)nch, n, sp_hist_quantile(h, 0.005), sp_hist_quantile(h, 0.25), med, sp_hist_quantile(h, 0.75), sp_hist_quantile(h, 0.995), mad, scale, mean, sd, vcut);
    fprintf(fj, "  \"hist\": [");
    for (int v = 0; v < 256; v++) fprintf(fj, "%llu%s", (unsigned long long)h[v], v < 255 ? ", " : "");
    fprintf(fj, "]\n}\n");
    fclose(fj);
    free(chs); free(S); free(cm); free(mk); free(val);
    if (zc != zl) z3_close(zc);
    z3_close(zl);
    return 0;
}
