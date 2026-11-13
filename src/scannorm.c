/* per-scan CT normalisation: see scannorm.h */
#include "scannorm.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SN_K = 48, SN_B = 32 };

int scan_norm_stats(z3 *(*open)(void *ctx, int level), void *ctx, int L, float *center, float *scale, int nthreads) {
    z3 *zl = open(ctx, L);
    if (!zl) return -2;
    const z3_meta *ml = z3_meta_of(zl);
    z3 *zc = nullptr, *zs = nullptr; int Lc = -1, Ls = -1; double ns = 0;
    for (int l = L + 2; l < L + 10 && l < 16; l++) {
        z3 *z = open(ctx, l); if (!z) continue;
        const z3_meta *m = z3_meta_of(z); const double n = (double)m->shape[0] * m->shape[1] * m->shape[2];
        if (n <= (double)(1 << 26)) { zc = z; Lc = l; break; }
        if (!zs || n < ns) { zs = z; Ls = l; ns = n; }
    }
    if (!zc && zs) { zc = zs; Lc = Ls; }   /* (a pyramid that stops early: its coarsest level, read subsampled) */
    if (!zc) return -3;
    const z3_meta *mc = z3_meta_of(zc);
    const int64_t full[3] = {mc->shape[0], mc->shape[1], mc->shape[2]};
    int st = 1; while ((double)full[0] * full[1] * full[2] / ((double)st * st * st) > (double)(1 << 26)) st *= 2;   /* every st-th voxel per axis */
    const int64_t nc[3] = {(full[0] + st - 1) / st, (full[1] + st - 1) / st, (full[2] + st - 1) / st};
    const size_t N = (size_t)nc[0] * nc[1] * nc[2];
    const int64_t sp = st * ((64 + st - 1) / st);   /* slab planes (a multiple of st) */
    uint8_t *cb = malloc(N), *bb = malloc((size_t)SN_B * SN_B * SN_B), *sl = st > 1 ? malloc((size_t)sp * full[1] * full[2]) : nullptr;
    if (!cb || !bb || (st > 1 && !sl)) { free(cb); free(bb); free(sl); return -4; }
    if (st == 1) { const int64_t o0[3] = {0, 0, 0}; if (z3_read(zc, o0, full, cb, nthreads)) { free(cb); free(bb); return -4; } }
    else for (int64_t z0 = 0; z0 < full[0]; z0 += sp) {   /* z slabs, subsampled into the compact grid */
        const int64_t o[3] = {z0, 0, 0}, n[3] = {z0 + sp < full[0] ? sp : full[0] - z0, full[1], full[2]};
        if (z3_read(zc, o, n, sl, nthreads)) { free(cb); free(bb); free(sl); return -4; }
        for (int64_t z = 0; z < n[0]; z += st) for (int64_t y = 0; y < full[1]; y += st) for (int64_t x = 0; x < full[2]; x += st)
            cb[((size_t)((z0 + z) / st) * nc[1] + y / st) * nc[2] + x / st] = sl[((size_t)z * full[1] + y) * full[2] + x];
    }
    free(sl);
    size_t nz = 0; for (size_t k = 0; k < N; k++) nz += cb[k] != 0;
    if (nz < SN_K) { free(cb); free(bb); return -5; }
    size_t want[SN_K]; for (int k = 0; k < SN_K; k++) want[k] = (size_t)(((2.0 * k + 1) * (double)nz) / (2.0 * SN_K));   /* fixed ranks, raster order */
    uint64_t h[256] = {0};
    const double f = (double)(1 << (Lc - L)) * st;
    size_t rank = 0; int k = 0;
    for (size_t q = 0; q < N && k < SN_K; q++) {
        if (!cb[q]) continue;
        if (rank++ != want[k]) continue;
        const int64_t c[3] = {(int64_t)(q / ((size_t)nc[1] * nc[2])), (int64_t)((q / nc[2]) % nc[1]), (int64_t)(q % nc[2])};
        int64_t o[3], n[3];
        for (int d = 0; d < 3; d++) {
            n[d] = ml->shape[d] < SN_B ? ml->shape[d] : SN_B;
            int64_t a = (int64_t)llround((c[d] + 0.5) * f) - n[d] / 2;
            o[d] = a < 0 ? 0 : a + n[d] > ml->shape[d] ? ml->shape[d] - n[d] : a;
        }
        if (z3_read(zl, o, n, bb, nthreads)) { free(cb); free(bb); return -6; }
        const size_t nb = (size_t)n[0] * n[1] * n[2];
        for (size_t i = 0; i < nb; i++) if (bb[i]) h[bb[i]]++;
        k++;
    }
    free(cb); free(bb);
    uint64_t tot = 0; double sum = 0, sq = 0; for (int v = 1; v < 256; v++) { tot += h[v]; sum += (double)v * h[v]; sq += (double)v * v * h[v]; }
    if (tot < 1000) return -7;
    /* mean / sd, not median / MAD: inside the scroll the nonzero voxels are bimodal (air in the mask, papyrus), and the MAD measures the
       dominant mode alone (1667 level 2: median 45, 1.4826 MAD 10.4 against a window sd of ~40): papyrus would sit at +7 */
    const double mean = sum / tot, sd = sqrt(fmax(sq / tot - mean * mean, 0));
    *center = (float)mean; *scale = (float)(sd > 0 ? sd : 1.0);
    return 0;
}

/* ---- scan_norm_stats_cached: per process (one entry per key, computed once, waiters block on it) and on disk ---- */
#include <errno.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
typedef struct sn_ent { char *key; int state, rc; float c, s; struct sn_ent *next; } sn_ent;   /* state 0 computing, 1 done */
static pthread_mutex_t sn_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t sn_cv = PTHREAD_COND_INITIALIZER; static sn_ent *sn_tab;
static int sn_key(z3 *(*open)(void *, int), void *ctx, int L, char *out, size_t n) {   /* the identity of what scan_norm_stats reads */
    size_t o = (size_t)snprintf(out, n, "scannorm v1 L %d", L);
    for (int l = L; l < L + 10 && l < 16; l++) {
        if (l == L + 1) continue;
        z3 *z = open(ctx, l); if (!z) { if (l == L) return -1; continue; }
        if (z3_base_of(z)) return -1;   /* (a derived array: its values are not the store's; no sharing, no disk) */
        const z3_meta *m = z3_meta_of(z);
        o += (size_t)snprintf(out + o, o < n ? n - o : 0, "|%d %s/%s %lld %lld %lld", l, z3_root(z), z3_key(z) ? z3_key(z) : "",
                              (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2]);
        if (l == L && z3_root(z)[0] == '/') {   /* (a local level L: its zarr.json's size / mtime, so a rewritten scan misses) */
            char p[2400]; struct stat st; snprintf(p, sizeof p, "%s/%s/zarr.json", z3_root(z), z3_key(z) ? z3_key(z) : "");
            if (!stat(p, &st)) o += (size_t)snprintf(out + o, o < n ? n - o : 0, " %lld %lld", (long long)st.st_size, (long long)st.st_mtime);
        }
    }
    return o < n ? 0 : -1;
}
static int sn_path(const char *key, char *out, size_t n) {
    if (getenv("UFSM_SCANNORM_CACHE") && !atoi(getenv("UFSM_SCANNORM_CACHE"))) return -1;
    const char *d = getenv("UFSM_SCANNORM_DIR"); char dd[1024];
    if (!d || !*d) { const char *h = getenv("HOME"); if (!h) return -1; snprintf(dd, sizeof dd, "%s/.cache/ufsm/scannorm", h); d = dd; }
    char tmp[1024]; snprintf(tmp, sizeof tmp, "%s", d);   /* mkdir -p */
    for (char *q = tmp + 1; *q; q++) if (*q == '/') { *q = 0; mkdir(tmp, 0755); *q = '/'; }
    if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
    unsigned char h[32]; unsigned nh = 0;
    if (!EVP_Digest(key, strlen(key), h, &nh, EVP_sha256(), nullptr) || nh != 32) return -1;
    size_t o = (size_t)snprintf(out, n, "%s/", d); for (int i = 0; i < 12 && o + 3 < n; i++) o += (size_t)snprintf(out + o, n - o, "%02x", h[i]);
    if (o + 5 < n) snprintf(out + o, n - o, ".txt"); else return -1;
    return 0;
}
static int sn_disk_get(const char *key, int *rc, float *c, float *s) {   /* the file: key line, then "rc cbits sbits" */
    char p[1200]; if (sn_path(key, p, sizeof p)) return -1;
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    size_t kl = strlen(key); char *ln = malloc(kl + 2); int ok = ln && fgets(ln, (int)kl + 2, f) && !strncmp(ln, key, kl) && ln[kl] == '\n';
    unsigned cb = 0, sb = 0; ok = ok && fscanf(f, "%d %x %x", rc, &cb, &sb) == 3;
    free(ln); fclose(f);
    if (!ok) return -1;
    memcpy(c, &cb, 4); memcpy(s, &sb, 4); return 0;
}
static void sn_disk_put(const char *key, int rc, float c, float s) {
    if (rc == -4 || rc == -6) return;   /* (read failures are not facts about the scan) */
    char p[1200], t[1300]; if (sn_path(key, p, sizeof p)) return;
    snprintf(t, sizeof t, "%s.%d.tmp", p, (int)getpid());
    FILE *f = fopen(t, "wb"); if (!f) return;
    unsigned cb, sb; memcpy(&cb, &c, 4); memcpy(&sb, &s, 4);
    fprintf(f, "%s\n%d %08x %08x\n", key, rc, cb, sb);
    if (fclose(f) || rename(t, p)) unlink(t);
}
int scan_norm_stats_cached(z3 *(*open)(void *ctx, int level), void *ctx, int L, float *center, float *scale, int nthreads, int *cached) {
    char key[4096];
    if (cached) *cached = 0;
    if (sn_key(open, ctx, L, key, sizeof key)) return scan_norm_stats(open, ctx, L, center, scale, nthreads);
    pthread_mutex_lock(&sn_mu);
    sn_ent *e = sn_tab; while (e && strcmp(e->key, key)) e = e->next;
    if (e) {
        while (!e->state) pthread_cond_wait(&sn_cv, &sn_mu);
        const int rc = e->rc; if (!rc) { *center = e->c; *scale = e->s; }
        pthread_mutex_unlock(&sn_mu); if (cached) *cached = 1; return rc;
    }
    e = calloc(1, sizeof *e); e->key = strdup(key); e->next = sn_tab; sn_tab = e;
    pthread_mutex_unlock(&sn_mu);
    int rc; float c = 0, s = 0; int hit = !sn_disk_get(key, &rc, &c, &s);
    if (!hit) { rc = scan_norm_stats(open, ctx, L, &c, &s, nthreads); sn_disk_put(key, rc, c, s); }
    pthread_mutex_lock(&sn_mu); e->rc = rc; e->c = c; e->s = s; e->state = 1; pthread_cond_broadcast(&sn_cv); pthread_mutex_unlock(&sn_mu);
    if (hit && getenv("UFSM_SCANNORM_VERIFY") && atoi(getenv("UFSM_SCANNORM_VERIFY"))) {   /* tests: the cached bits = a fresh computation */
        float c2 = 0, s2 = 0; const int r2 = scan_norm_stats(open, ctx, L, &c2, &s2, nthreads);
        if (r2 != rc || (!rc && (memcmp(&c2, &c, 4) || memcmp(&s2, &s, 4)))) { fprintf(stderr, "scannorm: the cached statistics differ from a fresh computation (%s)\n", key); abort(); }
        fprintf(stderr, "scannorm: verified %s\n", key);
    }
    if (!rc) { *center = c; *scale = s; }
    if (cached) *cached = hit;
    return rc;
}
