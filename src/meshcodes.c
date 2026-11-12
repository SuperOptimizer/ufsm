/* Winding codes on the fly from q-assigned surface meshes: see meshcodes.h. The stamp below is src/ingest.c raster_shard's --value path
   with the shard box replaced by the read box (the voxel of a sample is floor(p + 0.5) in both, independent of the box), same tile
   order, same sampling density (0.5 voxel along both quad edges at the target level, capped at 256). tests/test_codes_onfly.sh holds the
   two byte-identical. */
#include "meshcodes.h"
#include "meshcodesgpu.h"
#include "reg.h"
#include "store.h"
#include "tiff.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int w, h; float *xyz; uint8_t *valid; float *val; } mc_mesh;   /* xyz: level-0 x, y, z per vertex (after the map) */
typedef struct { int mi, r0, r1, c0, c1; float lo[3], hi[3]; } mc_tile;        /* bounds in level-0 z, y, x */
typedef struct { mc_mesh *m; int nm; mc_tile *t; size_t nt; double scale; void *pk; size_t pkn; char id[48]; } mc_ctx;   /* pk: the mmap'd pack (meshes + tiles point into it) */

static double mc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline uint8_t mc_code(double q) { long k = lround(q * 18.0) % 252; if (k < 0) k += 252; return (uint8_t)(1 + k); }
static inline int mc_diff(uint8_t a, uint8_t b) { int d = ((int)a - (int)b) % 252; if (d < -126) d += 252; if (d >= 126) d -= 252; return d; }

static float *mc_tif(const char *dir, const char *name, int *w, int *h) {
    char p[2200]; snprintf(p, sizeof p, "%s/%s", dir, name);
    tiff *t = tiff_open_file(p); if (!t) { fprintf(stderr, "mesh codes: %s: %s\n", p, tiff_error()); return nullptr; }
    tiff_page pg; float *v = nullptr;
    if (tiff_page_info(t, 0, &pg) || pg.w < 2 || pg.h < 2 || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 || (*w && (pg.w != *w || pg.h != *h)))
        fprintf(stderr, "mesh codes: %s: not a matching 2-D float32 image\n", p);
    else if (!(v = malloc((size_t)pg.w * pg.h * 4)) || tiff_read_page(t, 0, v)) { fprintf(stderr, "mesh codes: %s: TIFF decode failed\n", p); free(v); v = nullptr; }
    else { *w = pg.w; *h = pg.h; }
    tiff_close(t); return v;
}
static int mc_load(mc_mesh *m, const char *dir, const char *value, const rmap *map) {
    float *ax[3] = {mc_tif(dir, "x.tif", &m->w, &m->h), nullptr, nullptr};
    if (ax[0]) ax[1] = mc_tif(dir, "y.tif", &m->w, &m->h);
    if (ax[1]) ax[2] = mc_tif(dir, "z.tif", &m->w, &m->h);
    if (ax[2]) m->val = mc_tif(dir, value, &m->w, &m->h);
    const size_t n = (size_t)m->w * m->h;
    if (m->val) { m->xyz = malloc(n * 12); m->valid = malloc(n); }
    if (!m->xyz || !m->valid) { for (int a = 0; a < 3; a++) free(ax[a]); return -1; }
    for (size_t i = 0; i < n; i++) {   /* (validity on the meshes' own coordinates, as ufsm raster; then the map) */
        m->valid[i] = isfinite(ax[0][i]) && isfinite(ax[1][i]) && isfinite(ax[2][i]) && ax[2][i] > 0 && ax[0][i] >= 0;
        double c[3] = {ax[0][i], ax[1][i], ax[2][i]};
        if (map && m->valid[i]) { double f[3]; rmap_c2f(map, c, f); for (int a = 0; a < 3; a++) c[a] = f[a]; }
        for (int a = 0; a < 3; a++) m->xyz[i * 3 + a] = (float)c[a];
    }
    for (int a = 0; a < 3; a++) free(ax[a]);
    return 0;
}
static int mc_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void mc_free(void *vp) {
    mc_ctx *X = vp; if (!X) return;
    if (X->pk) munmap(X->pk, X->pkn);
    else { for (int i = 0; i < X->nm; i++) { free(X->m[i].xyz); free(X->m[i].valid); free(X->m[i].val); } free(X->t); }
    free(X->m); free(X);
}

/* The pack: the loaded meshes (vertices after the map, values, validity) and the tile index in one file, mmap'd read-only so an open
   costs milliseconds and every process shares the pages (Paris 4 June: 28 meshes, 5.8 GB of deflated TIFFs on the HDD took 228 s to
   load and ~7 GB of heap per process). <dir>/.meshcodes-<16 hex>.pack, or under $UFSM_MESHCODES_CACHE when set (a read-only meshes dir);
   UFSM_MESHCODES_CACHE=off: none. The key names every input (each mesh's four TIFFs with size and mtime, the value name, the map file
   with size and mtime, this layout's version); a pack whose stored key differs is rebuilt. Written to a temporary name and renamed.
   Layout (8-byte aligned): "UFMC0001", u64 key length, u64 meshes, u64 tiles, key, {i32 w, i32 h} per mesh, tiles, then per mesh xyz
   (12 n bytes), val (4 n), valid (n). */
#define MC_AL(x) (((x) + 7) & ~(size_t)7)
static void mc_key(char *k, size_t nk, const char *dir, char **names, int nn, const char *value, const char *mapp) {
    size_t u = (size_t)snprintf(k, nk, "meshcodes pack v1|value %s|", value); struct stat st;
    for (int i = 0; i < nn && u < nk; i++) { u += (size_t)snprintf(k + u, nk - u, "%s", names[i]);
        const char *f[4] = {"x.tif", "y.tif", "z.tif", value};
        for (int a = 0; a < 4 && u < nk; a++) { char p[2400]; snprintf(p, sizeof p, "%s/%s/%s", dir, names[i], f[a]);
            if (stat(p, &st)) memset(&st, 0, sizeof st);
            u += (size_t)snprintf(k + u, nk - u, ":%lld.%lld", (long long)st.st_size, (long long)st.st_mtime); }
        if (u < nk) u += (size_t)snprintf(k + u, nk - u, "|"); }
    if (mapp && *mapp && u < nk) { char rp[4200]; if (!realpath(mapp, rp)) snprintf(rp, sizeof rp, "%s", mapp);
        if (stat(rp, &st)) memset(&st, 0, sizeof st);
        char fp[4300]; snprintf(fp, sizeof fp, "%s", rp); char *sl = strrchr(fp, '/'); struct stat sf; long long fs = 0, fm = 0;   /* (the field beside reg.json) */
        if (sl) { snprintf(sl + 1, sizeof fp - (size_t)(sl + 1 - fp), "field.f32"); if (!stat(fp, &sf)) { fs = sf.st_size; fm = sf.st_mtime; } }
        snprintf(k + u, nk - u, "map %s:%lld.%lld:field %lld.%lld", rp, (long long)st.st_size, (long long)st.st_mtime, fs, fm); }
}
static uint64_t mc_hash(const char *s) { uint64_t h = 1469598103934665603ull; for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ull; } return h; }
static int mc_pack_path(char *out, size_t n, const char *dir, const char *key) {
    const char *e = getenv("UFSM_MESHCODES_CACHE"); if (e && (!strcmp(e, "off") || !strcmp(e, "0"))) return -1;
    snprintf(out, n, "%s/.meshcodes-%016llx.pack", e && *e ? e : dir, (unsigned long long)mc_hash(key)); return 0;
}
static int mc_pack_write(const mc_ctx *X, const char *path, const char *key) {
    char tmp[4400]; snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
    FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    const uint64_t hd[3] = {strlen(key), (uint64_t)X->nm, X->nt}; const char z8[8] = {0}; int ok = 1;
#define MC_W(p, n) do { const size_t n_ = (n); ok = ok && fwrite((p), 1, n_, f) == n_ && fwrite(z8, 1, MC_AL(n_) - n_, f) == MC_AL(n_) - n_; } while (0)
    MC_W("UFMC0001", 8); MC_W(hd, sizeof hd); MC_W(key, hd[0]);
    for (int i = 0; i < X->nm; i++) { const int32_t wh[2] = {X->m[i].w, X->m[i].h}; MC_W(wh, 8); }
    MC_W(X->t, X->nt * sizeof *X->t);
    for (int i = 0; i < X->nm; i++) { const size_t n = (size_t)X->m[i].w * X->m[i].h; MC_W(X->m[i].xyz, n * 12); MC_W(X->m[i].val, n * 4); MC_W(X->m[i].valid, n); }
#undef MC_W
    ok = !fclose(f) && ok;
    if (!ok || rename(tmp, path)) { unlink(tmp); return -1; }
    return 0;
}
static mc_ctx *mc_pack_open(const char *path, const char *key) {
    const int fd = open(path, O_RDONLY); if (fd < 0) return nullptr;
    struct stat st; if (fstat(fd, &st) || st.st_size < 40) { close(fd); return nullptr; }
    uint8_t *b = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0); close(fd); if (b == MAP_FAILED) return nullptr;
    const size_t N = (size_t)st.st_size, kl = strlen(key); uint64_t hd[3]; memcpy(hd, b + 8, sizeof hd); size_t u = 32;
    mc_ctx *X = nullptr;
    if (!memcmp(b, "UFMC0001", 8) && hd[0] == kl && u + MC_AL(kl) <= N && !memcmp(b + u, key, kl) && hd[1] > 0 && hd[1] < (1u << 20)) {
        u += MC_AL(kl); X = calloc(1, sizeof *X); X->nm = (int)hd[1]; X->nt = hd[2]; X->m = calloc(hd[1], sizeof *X->m); X->pk = b; X->pkn = N;
        int ok = u + 8 * hd[1] <= N;
        for (int i = 0; ok && i < X->nm; i++) { int32_t wh[2]; memcpy(wh, b + u + 8 * (size_t)i, 8); X->m[i].w = wh[0]; X->m[i].h = wh[1]; ok = wh[0] > 1 && wh[1] > 1; }
        u += 8 * hd[1]; ok = ok && X->nt <= (N - u) / sizeof *X->t;
        if (ok) { X->t = (mc_tile *)(b + u); u += MC_AL(X->nt * sizeof *X->t); }
        for (int i = 0; ok && i < X->nm; i++) { const size_t n = (size_t)X->m[i].w * X->m[i].h;
            if (u + MC_AL(n * 12) + MC_AL(n * 4) + MC_AL(n) > N) { ok = 0; break; }
            X->m[i].xyz = (float *)(b + u); u += MC_AL(n * 12); X->m[i].val = (float *)(b + u); u += MC_AL(n * 4); X->m[i].valid = b + u; u += MC_AL(n); }
        if (!ok) { free(X->m); free(X); X = nullptr; }
    }
    if (!X) munmap(b, N);
    return X;
}

/* one z range [z0, z1) of the read box; every thread walks the tiles in order and writes only its own slices: deterministic */
typedef struct { const mc_ctx *X; const int64_t *o, *n; uint8_t *out; int64_t z0, z1; } mc_part;
static void *mc_stamp(void *vp) {
    const mc_part *P = vp; const mc_ctx *X = P->X; const double sc = X->scale;
    const int64_t *o = P->o, *n = P->n;
    /* the quad / tile cull box: a sample in voxel g has p in [g - 0.5, g + 0.5), so anything outside [lo - 1, hi + 1] cannot hit */
    const double lo[3] = {(double)(o[0] + P->z0) - 1, (double)o[1] - 1, (double)o[2] - 1}, hi[3] = {(double)(o[0] + P->z1), (double)(o[1] + n[1]), (double)(o[2] + n[2])};
    for (size_t ti = 0; ti < X->nt; ti++) {
        const mc_tile *t = &X->t[ti]; int hit = 1;
        for (int d = 0; d < 3; d++) if (t->lo[d] * sc > hi[d] || t->hi[d] * sc < lo[d]) hit = 0;
        if (!hit) continue;
        const mc_mesh *m = &X->m[t->mi];
        for (int r = t->r0; r < t->r1; r++) for (int c = t->c0; c < t->c1; c++) {
            const size_t i00 = (size_t)r * m->w + c, idx[4] = {i00, i00 + 1, i00 + (size_t)m->w, i00 + (size_t)m->w + 1};
            if (!(m->valid[idx[0]] && m->valid[idx[1]] && m->valid[idx[2]] && m->valid[idx[3]])) continue;
            const double qv[4] = {m->val[idx[0]], m->val[idx[1]], m->val[idx[2]], m->val[idx[3]]};
            if (!(isfinite(qv[0]) && isfinite(qv[1]) && isfinite(qv[2]) && isfinite(qv[3]))) continue;
            double p[4][3], cl[3] = {1e30, 1e30, 1e30}, ch[3] = {-1e30, -1e30, -1e30};
            for (int k = 0; k < 4; k++) for (int a = 0; a < 3; a++) p[k][2 - a] = m->xyz[idx[k] * 3 + a] * sc;
            for (int k = 0; k < 4; k++) for (int d = 0; d < 3; d++) { if (p[k][d] < cl[d]) cl[d] = p[k][d]; if (p[k][d] > ch[d]) ch[d] = p[k][d]; }
            int outb = 0; for (int d = 0; d < 3; d++) if (cl[d] > hi[d] || ch[d] < lo[d]) outb = 1;
            if (outb) continue;
            double e1 = 0, e2 = 0;
            for (int d = 0; d < 3; d++) { e1 += (p[1][d] - p[0][d]) * (p[1][d] - p[0][d]); e2 += (p[2][d] - p[0][d]) * (p[2][d] - p[0][d]); }
            int nu = (int)ceil(sqrt(e1) / 0.5) + 1, nv = (int)ceil(sqrt(e2) / 0.5) + 1;
            if (nu > 256) nu = 256;
            if (nv > 256) nv = 256;
            for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
                const double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0;
                int64_t g[3]; int in = 1;
                for (int d = 0; d < 3; d++) {
                    const double q = (1 - u) * (1 - v) * p[0][d] + u * (1 - v) * p[1][d] + (1 - u) * v * p[2][d] + u * v * p[3][d];
                    g[d] = (int64_t)floor(q + 0.5) - o[d];
                    if (g[d] < (d ? 0 : P->z0) || g[d] >= (d ? n[d] : P->z1)) in = 0;
                }
                if (!in) continue;
                uint8_t *dv = &P->out[((size_t)g[0] * n[1] + g[1]) * n[2] + g[2]];
                const uint8_t code = mc_code((1 - u) * (1 - v) * qv[0] + u * (1 - v) * qv[1] + (1 - u) * v * qv[2] + u * v * qv[3]);
                if (*dv == 0) *dv = code;
                else if (*dv != 255 && abs(mc_diff(*dv, code)) > 4) *dv = 255;
            }
        }
    }
    return nullptr;
}
/* the device stamp (src/nn_mcodes.cu, byte-identical): the host culls the tiles against the box and gathers their vertex blocks (a tile:
   33 x 33 vertices, 17 bytes each), the device runs the samples. -1 undecided (the first read: $UFSM_MESHCODES_CPU=1 the CPU,
   $UFSM_MESHCODES_GPU the device, default 0), 0 the CPU, 1 the GPU. */
static _Atomic int g_mcgpu = -1;
static struct { _Atomic uint64_t gpu_reads, cpu_reads, gpu_fail, us_gpu, us_cpu; } g_mcs;
void mesh_codes_gpu_config(int dev) { if (dev < 0 || !mesh_codes_gpu_stamp) { g_mcgpu = 0; return; } mesh_codes_gpu_device(dev); g_mcgpu = 1; }
static int mc_gpu_on(void) {
    int g = atomic_load(&g_mcgpu); if (g >= 0) return g;
    const char *e = getenv("UFSM_MESHCODES_CPU"), *d = getenv("UFSM_MESHCODES_GPU");
    if ((e && atoi(e)) || !mesh_codes_gpu_stamp) g = 0; else { mesh_codes_gpu_device(d ? atoi(d) : 0); g = 1; }
    atomic_store(&g_mcgpu, g); return g;
}
void mesh_codes_stats(uint64_t *gpu_reads, uint64_t *cpu_reads, uint64_t *gpu_fail, double *gpu_s, double *cpu_s) {
    *gpu_reads = g_mcs.gpu_reads; *cpu_reads = g_mcs.cpu_reads; *gpu_fail = g_mcs.gpu_fail; *gpu_s = g_mcs.us_gpu * 1e-6; *cpu_s = g_mcs.us_cpu * 1e-6;
}
static thread_local struct { mcg_tile *t; size_t tc; float *xyz, *val; uint8_t *valid; size_t vc; } t_mg;
static int mc_read_gpu(const mc_ctx *X, const int64_t o[3], const int64_t n[3], uint8_t *out) {
    const double sc = X->scale, lo[3] = {(double)o[0] - 1, (double)o[1] - 1, (double)o[2] - 1}, hi[3] = {(double)(o[0] + n[0]), (double)(o[1] + n[1]), (double)(o[2] + n[2])};
    size_t nt = 0, nv = 0;
    for (size_t ti = 0; ti < X->nt; ti++) {   /* (mc_stamp's tile cull for the whole box; tiles keep their order) */
        const mc_tile *t = &X->t[ti]; int hit = 1;
        for (int d = 0; d < 3; d++) if (t->lo[d] * sc > hi[d] || t->hi[d] * sc < lo[d]) hit = 0;
        if (!hit) continue;
        const size_t k = (size_t)(t->r1 - t->r0 + 1) * (t->c1 - t->c0 + 1);
        if (nt == t_mg.tc) { const size_t c = t_mg.tc ? 2 * t_mg.tc : 1024; mcg_tile *nb = realloc(t_mg.t, c * sizeof *nb); if (!nb) return -1; t_mg.t = nb; t_mg.tc = c; }
        if (nv + k > t_mg.vc) { size_t c = t_mg.vc ? t_mg.vc : 1 << 16; while (c < nv + k) c *= 2;
            float *a = realloc(t_mg.xyz, c * 12); if (a) t_mg.xyz = a;
            float *b = realloc(t_mg.val, c * 4); if (b) t_mg.val = b;
            uint8_t *v = realloc(t_mg.valid, c); if (v) t_mg.valid = v;
            if (!a || !b || !v) return -1;
            t_mg.vc = c; }
        const mc_mesh *m = &X->m[t->mi]; const int nc = t->c1 - t->c0 + 1;
        t_mg.t[nt++] = (mcg_tile){.voff = (uint32_t)nv, .nr = t->r1 - t->r0 + 1, .nc = nc, .zlo = t->lo[0], .zhi = t->hi[0]};
        for (int r = t->r0; r <= t->r1; r++) { const size_t s = (size_t)r * m->w + t->c0;
            memcpy(t_mg.xyz + 3 * nv, m->xyz + 3 * s, (size_t)nc * 12); memcpy(t_mg.val + nv, m->val + s, (size_t)nc * 4); memcpy(t_mg.valid + nv, m->valid + s, (size_t)nc); nv += (size_t)nc; }
        if (nv > UINT32_MAX - 4096) return -1;
    }
    if (!nt) { memset(out, 0, (size_t)n[0] * n[1] * n[2]); return 0; }
    return mesh_codes_gpu_stamp(t_mg.t, (int)nt, t_mg.xyz, t_mg.val, t_mg.valid, nv, sc, o, n, out);
}
static int mc_read(void *vp, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; const mc_ctx *X = vp; const double t0 = mc_now();
    if (mc_gpu_on()) {
        if (!mc_read_gpu(X, o, n, out)) { atomic_fetch_add(&g_mcs.gpu_reads, 1); atomic_fetch_add(&g_mcs.us_gpu, (uint64_t)(1e6 * (mc_now() - t0))); return 0; }
        if (!atomic_fetch_add(&g_mcs.gpu_fail, 1)) fprintf(stderr, "mesh codes: the device stamp failed; this read (and any later failing one) runs on the CPU\n");
    }
    memset(out, 0, (size_t)n[0] * n[1] * n[2]);
    int T = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; if (T > n[0]) T = (int)n[0];
    mc_part P[64]; pthread_t th[64];
    for (int i = 0; i < T; i++) P[i] = (mc_part){X, o, n, out, n[0] * i / T, n[0] * (i + 1) / T};
    for (int i = 1; i < T; i++) if (pthread_create(&th[i], nullptr, mc_stamp, &P[i])) { mc_stamp(&P[i]); th[i] = 0; }
    mc_stamp(&P[0]);
    for (int i = 1; i < T; i++) if (th[i]) pthread_join(th[i], nullptr);
    atomic_fetch_add(&g_mcs.cpu_reads, 1); atomic_fetch_add(&g_mcs.us_cpu, (uint64_t)(1e6 * (mc_now() - t0)));
    return 0;
}
static int mc_extent(void *vp, const int64_t grid[3], const int chunk[3], uint8_t *bits) {
    const mc_ctx *X = vp;
    for (size_t ti = 0; ti < X->nt; ti++) { const mc_tile *t = &X->t[ti]; int64_t a[3], b[3]; int ok = 1;
        for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor((t->lo[d] * X->scale - 1) / chunk[d]); b[d] = (int64_t)floor((t->hi[d] * X->scale + 1) / chunk[d]);
            if (a[d] < 0) a[d] = 0;
            if (b[d] > grid[d] - 1) b[d] = grid[d] - 1;
            if (b[d] < a[d]) ok = 0; }
        if (ok) for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) bits[(z * grid[1] + y) * grid[2] + x] = 1; }
    return 0;
}

z3 *mesh_codes_open(const char *dir, const char *value, const char *mapp, int level, z3 *ct_level, int gpu) {
    if (gpu < 0) mesh_codes_gpu_config(-1); else if (gpu > 0) mesh_codes_gpu_config(gpu - 1);   /* (0: the process default) */
    if (!ct_level || !dir || !value || level < 0 || level > 8) { fprintf(stderr, "mesh codes: need a meshes dir, a value image name, level 0..8 and the CT level\n"); z3_close(ct_level); return nullptr; }
    const rmap *map = nullptr; if (mapp && *mapp && !(map = reg_map_load(mapp))) { fprintf(stderr, "mesh codes: cannot load the map %s\n", mapp); z3_close(ct_level); return nullptr; }
    char **names = nullptr; int nn = 0, cap = 0;
    DIR *D = opendir(dir); if (!D) { fprintf(stderr, "mesh codes: cannot open %s\n", dir); z3_close(ct_level); return nullptr; }
    for (struct dirent *e; (e = readdir(D));) { const size_t l = strlen(e->d_name);
        if (l < 8 || strcmp(e->d_name + l - 7, ".tifxyz")) continue;
        if (nn == cap) { cap = cap ? cap * 2 : 64; names = realloc(names, (size_t)cap * sizeof *names); }
        names[nn++] = strdup(e->d_name); }
    closedir(D);
    if (!nn) { fprintf(stderr, "mesh codes: no *.tifxyz in %s\n", dir); free(names); z3_close(ct_level); return nullptr; }
    qsort(names, (size_t)nn, sizeof *names, mc_cmp);   /* (the order of the shell glob ufsm raster was given) */
    const size_t nk = (size_t)nn * 400 + 9000; char *key = malloc(nk), pack[4300]; mc_key(key, nk, dir, names, nn, value, mapp);
    const int packed = !mc_pack_path(pack, sizeof pack, dir, key);
    mc_ctx *X = packed ? mc_pack_open(pack, key) : nullptr;
    const uint64_t kh = mc_hash(key);
    if (X) { for (int i = 0; i < nn; i++) free(names[i]);
        free(names); free(key); X->scale = 1.0 / (1 << level); snprintf(X->id, sizeof X->id, "meshcodes-v1-%016llx-L%d", (unsigned long long)kh, level);
        z3 *zp = z3_open_derived(ct_level, 0, mc_read, X, mc_free);
        if (!zp) { mc_free(X); z3_close(ct_level); return nullptr; }
        z3_derived_extent(zp, mc_extent, X); return zp; }
    X = calloc(1, sizeof *X); X->m = calloc((size_t)nn, sizeof *X->m); X->scale = 1.0 / (1 << level); snprintf(X->id, sizeof X->id, "meshcodes-v1-%016llx-L%d", (unsigned long long)kh, level);
    int bad = 0; size_t tcap = 0;
    for (int i = 0; i < nn && !bad; i++) { char p[2200]; snprintf(p, sizeof p, "%s/%s", dir, names[i]); bad = mc_load(&X->m[i], p, value, map); X->nm = i + 1; }
    for (int i = 0; i < nn; i++) free(names[i]);
    free(names);
    for (int mi = 0; mi < X->nm && !bad; mi++) { const mc_mesh *m = &X->m[mi];   /* the 32 x 32 tiles of ufsm raster's index, in its order */
        for (int r0 = 0; r0 + 1 < m->h; r0 += 32) for (int c0 = 0; c0 + 1 < m->w; c0 += 32) {
            mc_tile t = {.mi = mi, .r0 = r0, .c0 = c0, .r1 = r0 + 32, .c1 = c0 + 32, .lo = {1e30f, 1e30f, 1e30f}, .hi = {-1e30f, -1e30f, -1e30f}};
            if (t.r1 >= m->h) t.r1 = m->h - 1;
            if (t.c1 >= m->w) t.c1 = m->w - 1;
            int any = 0;
            for (int r = r0; r <= t.r1; r++) for (int c = c0; c <= t.c1; c++) { const size_t k = (size_t)r * m->w + c; if (!m->valid[k]) continue;
                any = 1;
                for (int a = 0; a < 3; a++) { const int d = 2 - a; const float v = m->xyz[k * 3 + a]; if (v < t.lo[d]) t.lo[d] = v; if (v > t.hi[d]) t.hi[d] = v; } }
            if (!any) continue;
            if (X->nt == tcap) { tcap = tcap ? tcap * 2 : 1024; X->t = realloc(X->t, tcap * sizeof *X->t); }
            X->t[X->nt++] = t;
        } }
    if (bad) { free(key); mc_free(X); z3_close(ct_level); return nullptr; }
    if (packed && mc_pack_write(X, pack, key)) fprintf(stderr, "mesh codes: could not write the pack %s (set UFSM_MESHCODES_CACHE to a writable directory); loading without it\n", pack);
    free(key);
    z3 *z = z3_open_derived(ct_level, 0, mc_read, X, mc_free);
    if (!z) { mc_free(X); z3_close(ct_level); return nullptr; }
    z3_derived_extent(z, mc_extent, X);
    return z;
}

const char *mesh_codes_key(const z3 *z) { const mc_ctx *X = z3_derived_ctx(z); return X ? X->id : nullptr; }

static const char *mc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
/* ufsm codes-onfly <meshes-dir> --ct ROOT KEY [--level 0] [--value q.tif] [--map reg.json] [--ref ROOT KEY] [--windows 8] [--P 256] [--threads 8]:
   the derived codes over windows drawn on the meshes (random tiles), ms per window, and the byte differences from a stored codes raster
   (`ufsm raster --value` at that level). KEY is the CT array of the level (its meta gives the grid). Exit 1 when a byte differs. */
int cmd_codes_onfly(int argc, char **argv) {
    int ci = 0, ri = 0; for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--ct") && i + 2 < argc) ci = i; if (!strcmp(argv[i], "--ref") && i + 2 < argc) ri = i; }
    if (argc < 3 || !ci) { fprintf(stderr, "usage: ufsm codes-onfly <meshes-dir> --ct ROOT KEY [--level 0] [--value q.tif] [--map reg.json] [--ref ROOT KEY] [--windows 8] [--P 256] [--threads 8] [--back COARSE_ROOT KEY0 [--slice OUT.pgm]]\n"); return 2; }
    const int level = atoi(mc_opt(argc, argv, "--level", "0")), nw = atoi(mc_opt(argc, argv, "--windows", "8")), P = atoi(mc_opt(argc, argv, "--P", "256")), T = atoi(mc_opt(argc, argv, "--threads", "8"));
    uint64_t seed = (uint64_t)atoll(mc_opt(argc, argv, "--seed", "1"));
    store *cs = store_open(argv[ci + 1]); z3 *c0 = cs ? z3_open(cs, argv[ci + 2], nullptr) : nullptr;
    if (!c0) { fprintf(stderr, "codes-onfly: cannot open the CT %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
    const double t0 = mc_now(); const int gd = atoi(mc_opt(argc, argv, "--gpu", "-1"));   /* --gpu D: the device stamp on GPU D (default -1: the CPU) */
    z3 *z = mesh_codes_open(argv[2], mc_opt(argc, argv, "--value", "q.tif"), mc_opt(argc, argv, "--map", nullptr), level, c0, gd < 0 ? -1 : gd + 1); if (!z) return 1;
    const mc_ctx *X = z3_derived_ctx(z); const int64_t *sh = z3_meta_of(z)->shape;
    printf("codes-onfly: key %s\n", mesh_codes_key(z));
    printf("codes-onfly: %d meshes, %zu tiles, opened in %.1f s (grid %lld %lld %lld)\n", X->nm, X->nt, mc_now() - t0, (long long)sh[0], (long long)sh[1], (long long)sh[2]);
    { int64_t g[3]; uint8_t *bits = nullptr; const int64_t nk = z3_chunk_map(z, g, &bits); free(bits);
      printf("codes-onfly: the extent: %lld of %lld chunks (%.1f%%)\n", (long long)nk, (long long)(g[0] * g[1] * g[2]), 100.0 * nk / (double)(g[0] * g[1] * g[2])); }
    z3 *ref = nullptr; if (ri) { store *rs = store_open(argv[ri + 1]); ref = rs ? z3_open(rs, argv[ri + 2], nullptr) : nullptr; if (!ref) { fprintf(stderr, "codes-onfly: cannot open %s %s\n", argv[ri + 1], argv[ri + 2]); return 1; } }
    long long ato[3] = {0, 0, 0}; const int at = sscanf(mc_opt(argc, argv, "--at", ""), "%lld,%lld,%lld", &ato[0], &ato[1], &ato[2]) == 3;   /* --at z,y,x: every window there */
    const size_t nv = (size_t)P * P * P; uint8_t *a = malloc(nv), *b = malloc(nv); long long bad = 0; double tsum = 0, tmax = 0;
    for (int w = 0; w < nw; w++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull; const mc_tile *t = &X->t[(seed >> 33) % X->nt];
        int64_t o[3], n[3] = {P, P, P};
        for (int d = 0; d < 3; d++) { seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            o[d] = (int64_t)(0.5 * (t->lo[d] + t->hi[d]) * X->scale) - P / 2 + (int64_t)((seed >> 33) % 33) - 16;   /* (not chunk aligned) */
            if (o[d] > sh[d] - P) o[d] = sh[d] - P;
            if (o[d] < 0) o[d] = 0; }
        if (at) for (int d = 0; d < 3; d++) o[d] = ato[d];
        const double t1 = mc_now(); if (z3_read(z, o, n, a, T)) { fprintf(stderr, "codes-onfly: read failed\n"); return 1; } const double dt = mc_now() - t1;
        tsum += dt; if (dt > tmax) tmax = dt;
        long long lab = 0, cf = 0, d = 0; for (size_t i = 0; i < nv; i++) { lab += a[i] != 0; cf += a[i] == 255; }
        if (ref) { if (z3_read(ref, o, n, b, 8)) { fprintf(stderr, "codes-onfly: ref read failed: %s\n", z3_error()); return 1; } for (size_t i = 0; i < nv; i++) d += a[i] != b[i]; }
        bad += d;
        printf("window %d (%lld,%lld,%lld): %.0f ms, %lld coded voxels (%lld conflict)", w, (long long)o[0], (long long)o[1], (long long)o[2], dt * 1e3, lab, cf);
        if (ref) printf(", %lld bytes differ", d);
        printf("\n"); fflush(stdout);
    }
    /* --back ROOT KEY (with --map): the map check. KEY = the coarse scan's level-0 CT array; every coded voxel of the last fine window
       goes back through rmap_f2c and must meet the coarse scan's own (unmapped) codes: the share with the same code (within 1 step =
       1/18 turn) at the nearest coarse voxel and within 1 / 2 voxels. --slice OUT.pgm: the window's middle z slice of the fine CT
       with the coded voxels drawn white (eye check). */
    int bi = 0; for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], "--back")) bi = i;
    const char *mapp = mc_opt(argc, argv, "--map", nullptr);
    if (bi && mapp && nw > 0) {
        const rmap *map = reg_map_load(mapp); store *bs = store_open(argv[bi + 1]); z3 *bc = bs ? z3_open(bs, argv[bi + 2], nullptr) : nullptr;
        z3 *zc = map && bc ? mesh_codes_open(argv[2], mc_opt(argc, argv, "--value", "q.tif"), nullptr, 0, bc, 0) : nullptr;
        if (!zc) { fprintf(stderr, "codes-onfly: --back: cannot open the map / the coarse CT\n"); return 1; }
        seed = (uint64_t)atoll(mc_opt(argc, argv, "--seed", "1"));   /* the first window again */
        seed = seed * 6364136223846793005ull + 1442695040888963407ull; const mc_tile *t = &X->t[(seed >> 33) % X->nt]; int64_t o[3], n[3] = {P, P, P};
        for (int d = 0; d < 3; d++) { seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            o[d] = (int64_t)(0.5 * (t->lo[d] + t->hi[d]) * X->scale) - P / 2 + (int64_t)((seed >> 33) % 33) - 16;
            if (o[d] > sh[d] - P) o[d] = sh[d] - P;
            if (o[d] < 0) o[d] = 0; }
        if (at) for (int d = 0; d < 3; d++) o[d] = ato[d];
        if (z3_read(z, o, n, a, T)) return 1;
        /* the coarse box holding the window's image */
        double clo[3] = {1e30, 1e30, 1e30}, chi[3] = {-1e30, -1e30, -1e30}; const double fs = 1.0 / X->scale;
        for (int c8 = 0; c8 < 8; c8++) { const double f[3] = {(o[2] + (c8 & 1 ? P : 0)) * fs, (o[1] + (c8 & 2 ? P : 0)) * fs, (o[0] + (c8 & 4 ? P : 0)) * fs}; double c[3]; rmap_f2c(map, f, c);
            for (int d = 0; d < 3; d++) { if (c[2 - d] < clo[d]) clo[d] = c[2 - d]; if (c[2 - d] > chi[d]) chi[d] = c[2 - d]; } }
        const int64_t *csh = z3_meta_of(zc)->shape; int64_t co[3], cn[3];
        for (int d = 0; d < 3; d++) { co[d] = (int64_t)floor(clo[d]) - 4; if (co[d] < 0) co[d] = 0; int64_t e = (int64_t)ceil(chi[d]) + 5; if (e > csh[d]) e = csh[d]; cn[d] = e > co[d] ? e - co[d] : 1; }
        uint8_t *cb = malloc((size_t)cn[0] * cn[1] * cn[2]); if (!cb || z3_read(zc, co, cn, cb, T)) { fprintf(stderr, "codes-onfly: --back: coarse read failed\n"); return 1; }
        long long tot = 0, hit[3] = {0, 0, 0}, none = 0;
        for (int64_t zz = 0; zz < P; zz++) for (int64_t y = 0; y < P; y++) for (int64_t x = 0; x < P; x++) { const uint8_t v = a[((size_t)zz * P + y) * P + x]; if (!v || v == 255) continue;
            const double f[3] = {(o[2] + x) * fs, (o[1] + y) * fs, (o[0] + zz) * fs}; double c[3]; rmap_f2c(map, f, c);
            const int64_t g[3] = {(int64_t)floor(c[2] + 0.5) - co[0], (int64_t)floor(c[1] + 0.5) - co[1], (int64_t)floor(c[0] + 0.5) - co[2]};
            tot++; int best = 9, any = 0;
            for (int dz = -2; dz <= 2; dz++) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) { const int64_t q[3] = {g[0] + dz, g[1] + dy, g[2] + dx};
                if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= cn[0] || q[1] >= cn[1] || q[2] >= cn[2]) continue;
                const uint8_t w8 = cb[((size_t)q[0] * cn[1] + q[1]) * cn[2] + q[2]]; if (!w8 || w8 == 255) continue;
                any = 1; if (abs(mc_diff(v, w8)) > 1) continue;
                const int r = abs(dz) > abs(dy) ? (abs(dz) > abs(dx) ? abs(dz) : abs(dx)) : (abs(dy) > abs(dx) ? abs(dy) : abs(dx)); if (r < best) best = r; }
            if (!any) none++;
            for (int r = 0; r < 3; r++) hit[r] += best <= r; }
        printf("codes-onfly: map check, fine window (%lld,%lld,%lld) %d^3 -> coarse box (%lld,%lld,%lld) %lldx%lldx%lld: %lld fine coded voxels; the same code back on the coarse scan at the nearest voxel %.4f, within 1 voxel %.4f, within 2 voxels %.4f; no coarse code within 2 voxels %.4f\n",
               (long long)o[0], (long long)o[1], (long long)o[2], P, (long long)co[0], (long long)co[1], (long long)co[2], (long long)cn[0], (long long)cn[1], (long long)cn[2], tot,
               tot ? (double)hit[0] / tot : 0, tot ? (double)hit[1] / tot : 0, tot ? (double)hit[2] / tot : 0, tot ? (double)none / tot : 0);
        const char *sl = mc_opt(argc, argv, "--slice", nullptr);
        if (sl) { store *fs2 = store_open(argv[ci + 1]); z3 *fc = fs2 ? z3_open(fs2, argv[ci + 2], nullptr) : nullptr; const int64_t so[3] = {o[0] + P / 2, o[1], o[2]}, sn[3] = {1, P, P};
            uint8_t *ct = malloc((size_t)P * P);
            if (fc && ct && !z3_read(fc, so, sn, ct, 8)) { FILE *f = fopen(sl, "wb"); if (f) { fprintf(f, "P5\n%d %d\n255\n", 2 * P, P);
                    for (int y = 0; y < P; y++) { fwrite(ct + (size_t)y * P, 1, (size_t)P, f);
                        for (int x = 0; x < P; x++) { const uint8_t v = a[((size_t)(P / 2) * P + y) * P + x]; fputc(v ? 255 : ct[(size_t)y * P + x] / 2, f); } }
                    fclose(f); printf("codes-onfly: slice z %lld -> %s (left the fine CT, right the CT dimmed with the coded voxels white)\n", (long long)so[0], sl); } }
            else fprintf(stderr, "codes-onfly: --slice: CT read failed\n");
            free(ct); z3_close(fc); }
        free(cb); z3_close(zc);
    }
    { uint64_t gr, cr, gf; double gs, cs; mesh_codes_stats(&gr, &cr, &gf, &gs, &cs); printf("codes-onfly: reads on the GPU %llu (%.2f s), on the CPU %llu (%.2f s), device failures %llu\n", (unsigned long long)gr, gs, (unsigned long long)cr, cs, (unsigned long long)gf); }
    printf("codes-onfly: %d windows of %d^3: %.0f ms mean, %.0f ms max per window (%s, %d threads)%s\n", nw, P, tsum / nw * 1e3, tmax * 1e3, gd >= 0 ? "GPU" : "CPU", T,
           ref ? (bad ? ": DIFFERS from the stored raster" : ": byte-identical to the stored raster") : "");
    free(a); free(b); z3_close(z); z3_close(ref);
    return bad ? 1 : 0;
}
