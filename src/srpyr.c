/* ufsm sr-pyrpairs <out-dir> --root R --key K [--um U] [--level 2] [--n 64] [--in 64] [--z Z0,Z1] [--minfrac 0.9] [--split train|test]
                    [--map reg.json --pair DIR --map-level 1 --cover-mm 1.2 --tol-um 2.4] [--threads 8] [--cache DIR] [--seed 1]
   SR pairs from one scan's own pyramid (same scan, same energy, exact alignment): input = level L (default 2: the 9.6 um rung of a 2.4 um
   scan) boxes of --in^3 voxels, target = the level-0 box under it ((4 in)^3 at 4x the origin; L must be 2 for x4). Box origins are drawn at
   random (seeded) with the box inside the level-0 z range --z (exclusive end; default the whole scan) and at least --minfrac non-zero input.
   --map: the target is instead a finer scan through a registration map (reg.json: coarse = this scan, fine = the finer scan, e.g. the
   1.13 um scan of the same scroll and energy), sampled trilinearly at its pyramid level --map-level (1: 2.26 um, ~ binned to the 2.4 um
   grid) at every level-0 voxel centre of the box (map evaluated on an 8-voxel lattice, trilinear in between); a target voxel is valid (> 0)
   when its stencil is; boxes need --minfrac valid targets and (with --pair) certified registration at the centre and corners.
   Output in the ufsm sr-pairs layout (src/srpair.h reads it): input.zarr / target.zarr (boxes stacked along z, lossless, 512^3 shards),
   boxes.json (every box --split), intensity_map.json (target 4^3 block mean -> input, pool-adjacent-violators). */
#include "reg.h"
#include "json.h"
#include "z3w.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

int reg_load_map(const char *path, const char *cache, rvol **cv, rvol **fv, rmap *m, rfield *D);   /* src/reg_view.c */

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

typedef struct {
    rvol *v, *fv; rmap m; rcover cov; int have_map, have_cov, L, B, T, ml, want; double minfrac; int64_t z0, z1;
    atomic_int next, nacc; int ntry; uint64_t seed;
    uint8_t *in, *tg; int64_t (*org)[3]; float *tfrac;   /* in: want x B^3, tg: want x T^3, org: level-L zyx origins */
    pthread_mutex_t mu; uint64_t hist[256][256];
} pjob;

static uint64_t mix64(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; return x ^ (x >> 31); }

/* target box through the map: level-0 voxel centres of the box -> fine level-0 (c2f) on an 8-voxel lattice -> fine level ml, trilinear */
static int map_target(pjob *j, const int64_t o0[3], uint8_t *tg) {
    const int T = j->T, G = T / 8 + 1; const double sc = 1.0 / (1 << j->ml);
    double *lat = malloc((size_t)G * G * G * 3 * sizeof *lat); if (!lat) return -1;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int c = 0; c < G; c++) for (int b = 0; b < G; b++) for (int a = 0; a < G; a++) {
        const double p[3] = {(double)o0[2] + 8 * a, (double)o0[1] + 8 * b, (double)o0[0] + 8 * c}; double f[3]; rmap_c2f(&j->m, p, f);
        double *q = lat + (((size_t)c * G + b) * G + a) * 3;
        for (int d = 0; d < 3; d++) { q[d] = (f[d] + 0.5) * sc - 0.5; lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); }
    }
    int64_t fo[3], fn[3]; for (int d = 0; d < 3; d++) { fo[2 - d] = (int64_t)floor(lo[d]) - 2; fn[2 - d] = (int64_t)ceil(hi[d]) + 2 - fo[2 - d] + 1; }
    if ((double)fn[0] * fn[1] * fn[2] > 200e6) { free(lat); return -1; }
    uint8_t *fb = malloc((size_t)fn[0] * fn[1] * fn[2]);
    if (!fb || rvol_read(j->fv, j->ml, fo, fn, fb)) { free(fb); free(lat); return -1; }
    for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {
        const int a = x >> 3, b = y >> 3, c = z >> 3; const double ua = (x & 7) / 8.0, ub = (y & 7) / 8.0, uc = (z & 7) / 8.0; double q[3] = {0, 0, 0};
        for (int k = 0; k < 8; k++) {
            const int aa = a + (k & 1), bb = b + (k >> 1 & 1), cc = c + (k >> 2); if (aa >= G || bb >= G || cc >= G) continue;
            const double w = ((k & 1) ? ua : 1 - ua) * ((k >> 1 & 1) ? ub : 1 - ub) * ((k >> 2) ? uc : 1 - uc);
            const double *l = lat + (((size_t)cc * G + bb) * G + aa) * 3; for (int d = 0; d < 3; d++) q[d] += w * l[d];
        }
        const double fx = q[0] - fo[2], fy = q[1] - fo[1], fz = q[2] - fo[0]; const int64_t ix = (int64_t)floor(fx), iy = (int64_t)floor(fy), iz = (int64_t)floor(fz);
        uint8_t *dst = tg + ((size_t)z * T + y) * T + x; *dst = 0;
        if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= fn[2] || iy + 1 >= fn[1] || iz + 1 >= fn[0]) continue;
        const double ux = fx - ix, uy = fy - iy, uz = fz - iz; double s = 0; int bad = 0;
        for (int k = 0; k < 8 && !bad; k++) { const uint8_t v = fb[((size_t)(iz + (k >> 2)) * fn[1] + iy + (k >> 1 & 1)) * fn[2] + ix + (k & 1)]; if (!v) bad = 1;
            s += ((k >> 2) ? uz : 1 - uz) * ((k >> 1 & 1) ? uy : 1 - uy) * ((k & 1) ? ux : 1 - ux) * v; }
        if (!bad) *dst = (uint8_t)fmin(255, fmax(1, s + 0.5));
    }
    free(fb); free(lat); return 0;
}

static void *pyr_worker(void *arg) {
    pjob *j = arg; const int B = j->B, T = j->T; const size_t NB = (size_t)B * B * B, NT = (size_t)T * T * T;
    uint8_t *in = malloc(NB), *tg = malloc(NT); uint64_t (*h)[256] = calloc(256, sizeof *h);
    const int64_t *s = j->v->shape[j->L], f = 1 << j->L;
    for (;;) {
        if (atomic_load(&j->nacc) >= j->want) break;
        const int k = atomic_fetch_add(&j->next, 1); if (k >= j->ntry) break;
        uint64_t r = mix64(j->seed * 0x9E3779B97F4A7C15ull + (uint64_t)k);
        int64_t o[3];   /* level-L zyx origin */
        const int64_t zlo = (j->z0 + f - 1) / f, zhi = j->z1 / f - B;
        if (zhi < zlo || s[1] < B || s[2] < B) break;
        o[0] = zlo + (int64_t)((r & 0xFFFFFFFF) % (uint64_t)(zhi - zlo + 1)); r = mix64(r);
        o[1] = (int64_t)((r & 0xFFFFFFFF) % (uint64_t)(s[1] - B + 1)); r = mix64(r);
        o[2] = (int64_t)((r & 0xFFFFFFFF) % (uint64_t)(s[2] - B + 1));
        const int64_t o0[3] = {4 * o[0], 4 * o[1], 4 * o[2]};
        if (j->have_cov) {   /* certified registration at the box centre and corners (this scan's level-0 xyz) */
            int cert = 1;
            for (int q = 0; q < 9 && cert; q++) {
                double p[3]; for (int d = 0; d < 3; d++) p[d] = q == 8 ? o0[2 - d] + T / 2.0 : o0[2 - d] + ((q >> d) & 1) * (T - 1);
                cert = rcover_ok(&j->cov, p);
            }
            if (!cert) continue;
        }
        const int64_t n[3] = {B, B, B};
        if (rvol_read(j->v, j->L, o, n, in)) continue;
        size_t nz = 0; for (size_t i = 0; i < NB; i++) nz += in[i] != 0;
        if (nz < j->minfrac * NB) continue;
        if (j->have_map) { if (map_target(j, o0, tg)) continue; }
        else { const int64_t nt[3] = {T, T, T}; if (rvol_read(j->v, 0, o0, nt, tg)) continue; }
        size_t nv = 0; for (size_t i = 0; i < NT; i++) nv += tg[i] != 0;
        if (nv < j->minfrac * NT) continue;
        for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
            const uint8_t iv = in[((size_t)z * B + y) * B + x]; if (!iv) continue;
            int sm = 0, ok = 1;
            for (int q = 0; q < 64 && ok; q++) { const uint8_t v = tg[((size_t)(4 * z + (q >> 4)) * T + 4 * y + (q >> 2 & 3)) * T + 4 * x + (q & 3)]; if (!v) ok = 0; sm += v; }
            if (ok) h[(sm + 32) / 64][iv]++;
        }
        const int slot = atomic_fetch_add(&j->nacc, 1);
        if (slot >= j->want) break;
        memcpy(j->in + (size_t)slot * NB, in, NB); memcpy(j->tg + (size_t)slot * NT, tg, NT);
        memcpy(j->org[slot], o, sizeof o); j->tfrac[slot] = (float)nv / NT;
        if ((slot + 1) % 8 == 0) fprintf(stderr, "  sr-pyrpairs: %d / %d boxes (%d tried)\n", slot + 1, j->want, atomic_load(&j->next));
    }
    pthread_mutex_lock(&j->mu); for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) j->hist[a][b] += h[a][b]; pthread_mutex_unlock(&j->mu);
    free(in); free(tg); free(h);
    return nullptr;
}

static int write_stack(const char *path, const uint8_t *boxes, int n, int e, const char *what) {
    const int S = 512, per = S / e; const int64_t shape[3] = {(int64_t)n * e, e, e};
    char at[256]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"sr-pyrpairs %s: boxes stacked along z\"}}", what);
    z3w *w = z3w_create(path, shape, S, 0, 0, at); if (!w) { fprintf(stderr, "sr-pyrpairs: %s\n", z3w_error()); return -1; }
    uint8_t *buf = calloc((size_t)S * S * S, 1); const size_t e3 = (size_t)e * e * e;
    for (int sh = 0; sh * per < n; sh++) {
        memset(buf, 0, (size_t)S * S * S);
        for (int q = 0; q < per && sh * per + q < n; q++) for (int z = 0; z < e; z++) for (int y = 0; y < e; y++)
            memcpy(buf + ((size_t)(q * e + z) * S + y) * S, boxes + (size_t)(sh * per + q) * e3 + ((size_t)z * e + y) * e, (size_t)e);
        if (z3w_write_shard(w, sh, 0, 0, buf, 8)) { fprintf(stderr, "sr-pyrpairs: %s\n", z3w_error()); free(buf); return -1; }
    }
    free(buf); return z3w_close(w);
}

int cmd_sr_pyrpairs(int argc, char **argv) {
    const char *root = opt(argc, argv, "--root", nullptr), *key = opt(argc, argv, "--key", nullptr);
    if (argc < 3 || argv[2][0] == '-' || !root || !key) {
        fprintf(stderr, "usage: ufsm sr-pyrpairs <out-dir> --root R --key K [--um U] [--level 2] [--n 64] [--in 64] [--z Z0,Z1] [--minfrac 0.9] [--split train|test]\n"
                        "       [--map reg.json --pair DIR --map-level 1 --cover-mm 1.2 --tol-um 2.4] [--threads 8] [--cache DIR] [--seed 1]\n");
        return 2;
    }
    const char *od = argv[2], *cache = opt(argc, argv, "--cache", nullptr);
    pjob *j = calloc(1, sizeof *j);
    j->L = atoi(opt(argc, argv, "--level", "2")); j->B = atoi(opt(argc, argv, "--in", "64")); j->T = 4 * j->B; j->want = atoi(opt(argc, argv, "--n", "64"));
    j->minfrac = atof(opt(argc, argv, "--minfrac", "0.9")); j->seed = (uint64_t)atoll(opt(argc, argv, "--seed", "1")); j->ml = atoi(opt(argc, argv, "--map-level", "1"));
    if (j->L != 2 || j->B < 8 || 512 % j->T || j->want < 1) { fprintf(stderr, "sr-pyrpairs: --level 2 (x4), --in with 4 in dividing 512\n"); return 2; }
    j->v = rvol_open(root, key, cache, atof(opt(argc, argv, "--um", "0")));
    if (!j->v || j->v->nl <= j->L) { fprintf(stderr, "sr-pyrpairs: cannot open %s/%s level %d\n", root, key, j->L); return 1; }
    const char *mp = opt(argc, argv, "--map", nullptr); rfield D = {0};
    if (mp) {
        if (reg_load_map(mp, cache, nullptr, &j->fv, &j->m, &D)) return 1;
        if (D.d) j->m.D = &D;
        if (j->fv->nl <= j->ml) { fprintf(stderr, "sr-pyrpairs: the mapped scan has no level %d\n", j->ml); return 1; }
        j->have_map = 1;
        const char *pd = opt(argc, argv, "--pair", nullptr);
        if (pd) { if (rcover_load(&j->cov, pd, atof(opt(argc, argv, "--cover-mm", "1.2")) * 1000 / j->v->um, atof(opt(argc, argv, "--tol-um", "2.4")) / j->v->um)) return 1; j->have_cov = 1; }
    }
    j->z0 = 0; j->z1 = j->v->shape[0][0];
    const char *zr = opt(argc, argv, "--z", nullptr);
    if (zr) { long long a, b; if (sscanf(zr, "%lld,%lld", &a, &b) != 2 || a < 0 || b <= a) { fprintf(stderr, "--z Z0,Z1 (level-0 voxels)\n"); return 2; } j->z0 = a; j->z1 = b < j->z1 ? b : j->z1; }
    j->ntry = 400 * j->want;
    const size_t NB = (size_t)j->B * j->B * j->B, NT = (size_t)j->T * j->T * j->T;
    j->in = malloc((size_t)j->want * NB); j->tg = malloc((size_t)j->want * NT); j->org = calloc((size_t)j->want, sizeof *j->org); j->tfrac = calloc((size_t)j->want, sizeof *j->tfrac);
    if (!j->in || !j->tg) { fprintf(stderr, "sr-pyrpairs: out of memory\n"); return 1; }
    pthread_mutex_init(&j->mu, nullptr);
    const time_t t0 = time(nullptr);
    fprintf(stderr, "sr-pyrpairs: %s/%s level %d (%.3f um) %d^3 -> %s %d^3, z %lld .. %lld\n", root, key, j->L, j->v->um * (1 << j->L), j->B,
            mp ? "mapped scan" : "level 0", j->T, (long long)j->z0, (long long)j->z1);
    const int nt = atoi(opt(argc, argv, "--threads", "8")); pthread_t th[64]; const int TT = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < TT; t++) pthread_create(&th[t], nullptr, pyr_worker, j);
    for (int t = 0; t < TT; t++) pthread_join(th[t], nullptr);
    int n = atomic_load(&j->nacc); if (n > j->want) n = j->want;
    if (n < 1) { fprintf(stderr, "sr-pyrpairs: no box accepted\n"); return 1; }
    mkdir(od, 0755);
    char p[2200];
    snprintf(p, sizeof p, "%s/input.zarr", od); if (write_stack(p, j->in, n, j->B, "input (pyramid level)")) return 1;
    snprintf(p, sizeof p, "%s/target.zarr", od); if (write_stack(p, j->tg, n, j->T, mp ? "target (mapped scan)" : "target (level 0)")) return 1;
    const char *split = opt(argc, argv, "--split", "train");
    snprintf(p, sizeof p, "%s/boxes.json", od); FILE *f = fopen(p, "w");
    fprintf(f, "{\"source\":\"sr-pyrpairs\",\"input\":{\"key\":\"%s\",\"level\":%d,\"um\":%.6g,\"box\":%d},\"target\":{\"key\":\"%s\",\"um\":%.6g,\"box\":%d,\"scale\":4%s%s%s},\"n\":%d,\"requested\":%d,\"z\":[%lld,%lld],\n \"boxes\":[",
            key, j->L, j->v->um * (1 << j->L), j->B, mp ? j->fv->name : key, j->v->um, j->T, mp ? ",\"map\":\"" : "", mp ? mp : "", mp ? "\"" : "", n, j->want, (long long)j->z0, (long long)j->z1);
    for (int i = 0; i < n; i++) fprintf(f, "%s\n  {\"slot\":%d,\"origin_level_zyx\":[%lld,%lld,%lld],\"origin_l0_zyx\":[%lld,%lld,%lld],\"target_valid\":%.3f,\"split\":\"%s\"}", i ? "," : "", i,
                                        (long long)j->org[i][0], (long long)j->org[i][1], (long long)j->org[i][2], (long long)(4 * j->org[i][0]), (long long)(4 * j->org[i][1]), (long long)(4 * j->org[i][2]), j->tfrac[i], split);
    fprintf(f, "\n ]}\n"); fclose(f);
    /* intensity map (as ufsm sr-pairs): per target bin the mean input, pool-adjacent-violators */
    double w[256], y[256]; for (int a = 0; a < 256; a++) { double c = 0, s = 0; for (int b = 1; b < 256; b++) { c += j->hist[a][b]; s += (double)b * j->hist[a][b]; } w[a] = c; y[a] = c > 0 ? s / c : 0; }
    double bw[256], bv[256]; int bs[256], nb = 0;
    for (int a = 0; a < 256; a++) { if (w[a] <= 0) continue; bv[nb] = y[a]; bw[nb] = w[a]; bs[nb] = a; nb++;
        while (nb > 1 && bv[nb - 2] > bv[nb - 1]) { bv[nb - 2] = (bv[nb - 2] * bw[nb - 2] + bv[nb - 1] * bw[nb - 1]) / (bw[nb - 2] + bw[nb - 1]); bw[nb - 2] += bw[nb - 1]; nb--; } }
    snprintf(p, sizeof p, "%s/intensity_map.json", od); f = fopen(p, "w");
    fprintf(f, "{\"maps\":\"target (4^3 block mean) -> input intensity\",\"method\":\"bin means + pool adjacent violators\",\"lut\":[");
    int k2 = 0; for (int a = 0; a < 256; a++) { while (k2 + 1 < nb && bs[k2 + 1] <= a) k2++; fprintf(f, "%s%.3f", a ? "," : "", nb ? bv[k2] : (double)a); }
    fprintf(f, "],\"bin_counts\":["); for (int a = 0; a < 256; a++) fprintf(f, "%s%.0f", a ? "," : "", w[a]); fprintf(f, "]}\n"); fclose(f);
    fprintf(stderr, "sr-pyrpairs: %d boxes from %d tries in %lds -> %s\n", n, atomic_load(&j->next), (long)(time(nullptr) - t0), od);
    return 0;
}
