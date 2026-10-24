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
#include <stdint.h>
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

/* ufsm sr-ceiling <pair-dir> --root R --key K [--um U] [--lut 1] [--cache DIR] [--max 64]: for an sr-pyrpairs set, the scan's own level-0
   box at each box origin scored against the set's target with the restore-eval-pairs metrics (PSNR, SSIM over 8^3 blocks, gradient PSNR;
   voxels where both are > 0; the target through the set's intensity map with --lut 1): the "perfect 2.4 um prediction" ceiling for a
   finer-scan target. Also the level-0 box against its own 3^3 box mean (a light denoise: a rough noise-floor indicator). */
#include "srpair.h"
typedef struct { double n, se, gn, gse, ss, nb; } cmet;
static void cmet_add(cmet *m, const uint8_t *a, const uint8_t *t, int T) {   /* a: prediction, t: target */
    const size_t str[3] = {(size_t)T * T, (size_t)T, 1};
    for (size_t i = 0; i < (size_t)T * T * T; i++) {
        if (!a[i] || !t[i]) continue;
        const double e = (double)a[i] - t[i]; m->n++; m->se += e * e;
        for (int d = 0; d < 3; d++) { if ((int)((i / str[d]) % T) == T - 1) continue; const size_t j = i + str[d]; if (!a[j] || !t[j]) continue;
            const double g = ((double)a[j] - a[i]) - ((double)t[j] - t[i]); m->gn++; m->gse += g * g; }
    }
    for (int Z = 0; Z < T / 8; Z++) for (int Y = 0; Y < T / 8; Y++) for (int X = 0; X < T / 8; X++) {
        double s[8] = {0}; int ok = 1;
        for (int z = 0; z < 8 && ok; z++) for (int y = 0; y < 8 && ok; y++) for (int x = 0; x < 8; x++) {
            const size_t i = ((size_t)(8 * Z + z) * T + 8 * Y + y) * T + 8 * X + x; if (!a[i] || !t[i]) { ok = 0; break; }
            s[0] += a[i]; s[1] += t[i]; s[2] += (double)a[i] * a[i]; s[3] += (double)t[i] * t[i]; s[4] += (double)a[i] * t[i];
        }
        if (!ok) continue;
        const double N = 512, ma = s[0] / N, mt = s[1] / N, va = s[2] / N - ma * ma, vt = s[3] / N - mt * mt, c = s[4] / N - ma * mt, C1 = 6.5025, C2 = 58.5225;
        m->ss += (2 * ma * mt + C1) * (2 * c + C2) / ((ma * ma + mt * mt + C1) * (va + vt + C2)); m->nb++;
    }
}
static void cmet_print(const char *what, const cmet *m) {
    printf("%-34s psnr %.3f  ssim %.4f  grad psnr %.3f  (%.0f voxels, %.0f blocks)\n", what, m->se > 0 ? 10 * log10(65025.0 * m->n / m->se) : 99.0,
           m->nb > 0 ? m->ss / m->nb : 0, m->gse > 0 ? 10 * log10(65025.0 * m->gn / m->gse) : 99.0, m->n, m->nb);
}
int cmd_sr_ceiling(int argc, char **argv) {
    const char *root = opt(argc, argv, "--root", nullptr), *key = opt(argc, argv, "--key", nullptr);
    if (argc < 3 || argv[2][0] == '-' || !root || !key) { fprintf(stderr, "usage: ufsm sr-ceiling <pair-dir> --root R --key K [--um U] [--lut 1] [--cache DIR] [--max 64]\n"); return 2; }
    srpair *p = srpair_open(argv[2], "val", atoi(opt(argc, argv, "--lut", "1")), 0); if (!p) return 1;
    rvol *v = rvol_open(root, key, opt(argc, argv, "--cache", nullptr), atof(opt(argc, argv, "--um", "0"))); if (!v) return 1;
    char bp[2100]; snprintf(bp, sizeof bp, "%s/boxes.json", argv[2]);
    FILE *f = fopen(bp, "rb"); if (!f) return 1; fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(L + 1); if (fread(t, 1, L, f) != (size_t)L) { fclose(f); return 1; } fclose(f); t[L] = 0;
    json *j = json_parse(t, (size_t)L); free(t); const json *bx = j ? json_get(j, "boxes") : nullptr; if (!bx) return 1;
    const int T = p->T, mx = atoi(opt(argc, argv, "--max", "64")); const size_t t3 = (size_t)T * T * T;
    uint8_t *tg = malloc(t3), *co = malloc((size_t)p->B * p->B * p->B), *a = malloc(t3), *dn = malloc(t3);
    cmet real = {0}, den = {0};
    for (size_t k = 0; k < bx->n && (int)k < mx; k++) {
        const json *b = json_at(bx, k); int64_t o[3]; for (int d = 0; d < 3; d++) o[d] = (int64_t)json_num(json_at(json_get(b, "origin_l0_zyx"), (size_t)d), -1);
        if (o[0] < 0 && json_num(json_get(json_get(j, "input"), "level"), -1) == 2)   /* sr-xpairs sets whose input is level 2 of this scan */
            for (int d = 0; d < 3; d++) o[d] = 4 * (int64_t)json_num(json_at(json_get(b, "origin_level_zyx"), (size_t)d), -1);
        const int oc[3] = {0, 0, 0};
        if (o[0] < 0 || srpair_read(p, (int)json_num(json_get(b, "slot"), (double)k), oc, T, tg, co)) continue;
        const int64_t n[3] = {T, T, T}; if (rvol_read(v, 0, o, n, a)) continue;
        cmet_add(&real, a, tg, T);
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {   /* 3^3 box mean of a (valid voxels) */
            double s = 0; int c = 0;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const int zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= T || yy >= T || xx >= T) continue;
                const uint8_t q = a[((size_t)zz * T + yy) * T + xx]; if (q) { s += q; c++; } }
            dn[((size_t)z * T + y) * T + x] = a[((size_t)z * T + y) * T + x] && c ? (uint8_t)lrint(s / c) : 0;
        }
        cmet_add(&den, a, dn, T);
    }
    cmet_print("level 0 vs target", &real);
    cmet_print("level 0 vs its 3^3 box mean", &den);
    json_free(j); free(tg); free(co); free(a); free(dn); srpair_free(p);
    return 0;
}

/* ufsm sr-xpairs <out-dir> --input ROOT,KEY,UM,LEVEL --target ROOT,KEY,UM,LEVEL --chain MAP:DIR:PAIRDIR[;MAP:DIR:PAIRDIR...] [--n 64] [--in 64]
                  [--z Z0,Z1 (input level-0 z of box origins)] [--holdout 0.15] [--minfrac 0.9] [--cover-mm 1.2] [--tol-um 2.4] [--threads 8]
                  [--cache DIR] [--seed 1]
   Cross-scan SR pairs in the coarse-box layout of ufsm sr-pairs (src/srpair.h): input = the raw --in^3 box of the input scan at its level,
   target = the target scan (any finer scan: e.g. a 1.13 um scan at level 1, a 0.55 um scan at level 2, ~2.2-2.3 um) sampled trilinearly
   through a chain of registration maps at the input box's x4 grid (target voxel t at input-level coords origin + (t + 0.5) / 4 - 0.5), i.e. the
   finer scan binned to ~input / 4. Each chain step is a reg.json applied from the input side ("c2f": the point is in the map's coarse frame,
   "f2c": in its fine frame), certified by its pair dir (a good fit box near, no failing evaluation box) at the box centre and corners; the chain
   is evaluated on an 8-target-voxel lattice and interpolated. Boxes need --minfrac non-zero input and valid target; the top --holdout of the
   accepted boxes by z is "test"; candidates are drawn around random points of the target scan mapped back through the chain (the overlap).
   intensity_map.json: target 4^3 block mean -> input (pool-adjacent-violators). */
typedef struct { rmap m; rfield D; int c2f; rcover cov; } xstep;
typedef struct {
    rvol *vi, *vt; int Li, Lt, B, T, want, ns, same; xstep st[4]; double minfrac, zcut; int64_t z0, z1;
    atomic_int next, nacc; int ntry; uint64_t seed;
    uint8_t *in, *tg; int64_t (*org)[3]; float *tfrac; int8_t *split;
    pthread_mutex_t mu; uint64_t hist[256][256];
} xjob;
/* input level-0 xyz -> target level-0 xyz through the chain; certified (every step's cover) when cert != nullptr */
static void x_map(const xjob *j, const double p[3], double q[3], int *cert) {
    double a[3] = {p[0], p[1], p[2]};
    for (int k = 0; k < j->ns; k++) {
        const xstep *t = &j->st[k]; double b[3];
        if (t->c2f) { if (cert) *cert &= rcover_ok(&t->cov, a); rmap_c2f(&t->m, a, b); }
        else { rmap_f2c(&t->m, a, b); if (cert) *cert &= rcover_ok(&t->cov, b); }
        memcpy(a, b, sizeof a);
    }
    memcpy(q, a, sizeof a);
}
/* target level-0 xyz -> input level-0 xyz (the chain backwards) */
static void x_unmap(const xjob *j, const double p[3], double q[3]) {
    double a[3] = {p[0], p[1], p[2]};
    for (int k = j->ns - 1; k >= 0; k--) { const xstep *t = &j->st[k]; double b[3]; if (t->c2f) rmap_f2c(&t->m, a, b); else rmap_c2f(&t->m, a, b); memcpy(a, b, sizeof a); }
    memcpy(q, a, sizeof a);
}
static void *x_worker(void *arg) {
    xjob *j = arg; const int B = j->B, T = j->T, G = T / 8 + 1; const size_t NB = (size_t)B * B * B, NT = (size_t)T * T * T;
    uint8_t *in = malloc(NB), *tg = malloc(NT); double *lat = malloc((size_t)G * G * G * 3 * sizeof *lat); uint64_t (*h)[256] = calloc(256, sizeof *h);
    const int64_t *s = j->vi->shape[j->Li]; const int64_t fi = (int64_t)1 << j->Li; const double ft = 1.0 / (1 << j->Lt);
    for (;;) {
        if (atomic_load(&j->nacc) >= j->want) break;
        const int k = atomic_fetch_add(&j->next, 1); if (k >= j->ntry) break;
        uint64_t r = mix64(j->seed * 0x9E3779B97F4A7C15ull + (uint64_t)k);
        const int64_t zlo = (j->z0 + fi - 1) / fi, zhi = j->z1 / fi - B; if (zhi < zlo || s[1] < B || s[2] < B) break;
        int64_t o[3];   /* input-level zyx origin: around a random point of the target scan mapped back (the overlap only) */
        { const int64_t *ts = j->vt->shape[0]; double tp[3], ip[3];
          for (int d = 0; d < 3; d++) { tp[d] = (double)((r & 0xFFFFFFFF) % (uint64_t)ts[2 - d]); r = mix64(r); }
          x_unmap(j, tp, ip);
          for (int d = 0; d < 3; d++) o[2 - d] = (int64_t)llround((ip[d] + 0.5) / fi - 0.5) - B / 2; }
        if (o[0] < zlo || o[0] > zhi || o[1] < 0 || o[2] < 0 || o[1] > s[1] - B || o[2] > s[2] - B) continue;
        int cert = 1;   /* centre and corners, input level-0 xyz */
        for (int q = 0; q < 9 && cert; q++) { double p[3], qq[3];
            for (int d = 0; d < 3; d++) { const double c = q == 8 ? o[2 - d] + B / 2.0 : o[2 - d] + ((q >> d) & 1) * (B - 1); p[d] = (c + 0.5) * fi - 0.5; }
            x_map(j, p, qq, &cert); }
        if (!cert) continue;
        const int64_t n[3] = {B, B, B};
        if (rvol_read(j->vi, j->Li, o, n, in)) continue;
        size_t nz = 0; for (size_t i = 0; i < NB; i++) nz += in[i] != 0;
        if (nz < j->minfrac * NB) continue;
        /* the chain on the lattice: target voxel (8a, 8b, 8c) -> target-level coords */
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < G; c++) for (int b = 0; b < G; b++) for (int a = 0; a < G; a++) {
            const int tv[3] = {8 * a, 8 * b, 8 * c}; double p[3], f[3];
            for (int d = 0; d < 3; d++) p[d] = ((o[2 - d] + (tv[d] + 0.5) / 4.0 - 0.5) + 0.5) * fi - 0.5;
            x_map(j, p, f, nullptr);
            double *qq = lat + (((size_t)c * G + b) * G + a) * 3;
            for (int d = 0; d < 3; d++) { qq[d] = (f[d] + 0.5) * ft - 0.5; lo[d] = fmin(lo[d], qq[d]); hi[d] = fmax(hi[d], qq[d]); }
        }
        int64_t fo[3], fn[3]; for (int d = 0; d < 3; d++) { fo[2 - d] = (int64_t)floor(lo[d]) - 2; fn[2 - d] = (int64_t)ceil(hi[d]) + 2 - fo[2 - d] + 1; }
        if ((double)fn[0] * fn[1] * fn[2] > 200e6) continue;
        uint8_t *fb = malloc((size_t)fn[0] * fn[1] * fn[2]);
        if (!fb || rvol_read(j->vt, j->Lt, fo, fn, fb)) { free(fb); continue; }
        size_t nv = 0;
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {
            const int a = x >> 3, b = y >> 3, c = z >> 3; const double ua = (x & 7) / 8.0, ub = (y & 7) / 8.0, uc = (z & 7) / 8.0; double q[3] = {0, 0, 0};
            for (int kk = 0; kk < 8; kk++) {
                int aa = a + (kk & 1), bb = b + (kk >> 1 & 1), cc = c + (kk >> 2); if (aa >= G) aa = G - 1; if (bb >= G) bb = G - 1; if (cc >= G) cc = G - 1;
                const double w = ((kk & 1) ? ua : 1 - ua) * ((kk >> 1 & 1) ? ub : 1 - ub) * ((kk >> 2) ? uc : 1 - uc);
                const double *l = lat + (((size_t)cc * G + bb) * G + aa) * 3; for (int d = 0; d < 3; d++) q[d] += w * l[d];
            }
            const double fx = q[0] - fo[2], fy = q[1] - fo[1], fz = q[2] - fo[0]; const int64_t ix = (int64_t)floor(fx), iy = (int64_t)floor(fy), iz = (int64_t)floor(fz);
            uint8_t *dst = tg + ((size_t)z * T + y) * T + x; *dst = 0;
            if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= fn[2] || iy + 1 >= fn[1] || iz + 1 >= fn[0]) continue;
            const double ux = fx - ix, uy = fy - iy, uz = fz - iz; double sv = 0; int bad = 0;
            for (int kk = 0; kk < 8 && !bad; kk++) { const uint8_t v = fb[((size_t)(iz + (kk >> 2)) * fn[1] + iy + (kk >> 1 & 1)) * fn[2] + ix + (kk & 1)]; if (!v) bad = 1;
                sv += ((kk >> 2) ? uz : 1 - uz) * ((kk >> 1 & 1) ? uy : 1 - uy) * ((kk & 1) ? ux : 1 - ux) * v; }
            if (!bad) { *dst = (uint8_t)fmin(255, fmax(1, sv + 0.5)); nv++; }
        }
        free(fb);
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
        if ((slot + 1) % 8 == 0) fprintf(stderr, "  sr-xpairs: %d / %d boxes (%d tried)\n", slot + 1, j->want, atomic_load(&j->next));
    }
    pthread_mutex_lock(&j->mu); for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) j->hist[a][b] += h[a][b]; pthread_mutex_unlock(&j->mu);
    free(in); free(tg); free(lat); free(h);
    return nullptr;
}
/* --same-grid: target = the finer scan's raw T^3 box at its level, input = the coarser scan resampled through the chain onto those voxels */
static void *x_worker_same(void *arg) {
    xjob *j = arg; const int T = j->T, G = T / 8 + 1; const size_t NT = (size_t)T * T * T;
    uint8_t *in = malloc(NT), *tg = malloc(NT); double *lat = malloc((size_t)G * G * G * 3 * sizeof *lat); uint64_t (*h)[256] = calloc(256, sizeof *h);
    const int64_t *ts = j->vt->shape[j->Lt]; const double ft = (double)(1 << j->Lt), fi = 1.0 / (1 << j->Li);
    for (;;) {
        if (atomic_load(&j->nacc) >= j->want) break;
        const int k = atomic_fetch_add(&j->next, 1); if (k >= j->ntry) break;
        uint64_t r = mix64(j->seed * 0x9E3779B97F4A7C15ull + (uint64_t)k);
        if (ts[0] < T || ts[1] < T || ts[2] < T) break;
        int64_t o[3]; for (int d = 0; d < 3; d++) { o[d] = (int64_t)((r & 0xFFFFFFFF) % (uint64_t)(ts[d] - T + 1)); r = mix64(r); }   /* target-level zyx */
        int cert = 1;
        for (int q = 0; q < 9 && cert; q++) { double tp[3], ip[3], qq[3];
            for (int d = 0; d < 3; d++) { const double c = q == 8 ? o[2 - d] + T / 2.0 : o[2 - d] + ((q >> d) & 1) * (T - 1); tp[d] = (c + 0.5) * ft - 0.5; }
            x_unmap(j, tp, ip); x_map(j, ip, qq, &cert); }
        if (!cert) continue;
        const int64_t n[3] = {T, T, T};
        if (rvol_read(j->vt, j->Lt, o, n, tg)) continue;
        size_t nv = 0; for (size_t i = 0; i < NT; i++) nv += tg[i] != 0;
        if (nv < j->minfrac * NT) continue;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < G; c++) for (int b = 0; b < G; b++) for (int a = 0; a < G; a++) {
            const int tv[3] = {8 * a, 8 * b, 8 * c}; double tp[3], ip[3];
            for (int d = 0; d < 3; d++) tp[d] = (o[2 - d] + tv[d] + 0.5) * ft - 0.5;
            x_unmap(j, tp, ip);
            double *qq = lat + (((size_t)c * G + b) * G + a) * 3;
            for (int d = 0; d < 3; d++) { qq[d] = (ip[d] + 0.5) * fi - 0.5; lo[d] = fmin(lo[d], qq[d]); hi[d] = fmax(hi[d], qq[d]); }
        }
        int64_t fo[3], fn[3]; for (int d = 0; d < 3; d++) { fo[2 - d] = (int64_t)floor(lo[d]) - 2; fn[2 - d] = (int64_t)ceil(hi[d]) + 2 - fo[2 - d] + 1; }
        if ((double)fn[0] * fn[1] * fn[2] > 200e6) continue;
        uint8_t *fb = malloc((size_t)fn[0] * fn[1] * fn[2]);
        if (!fb || rvol_read(j->vi, j->Li, fo, fn, fb)) { free(fb); continue; }
        size_t nz = 0;
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {
            const int a = x >> 3, b = y >> 3, c = z >> 3; const double ua = (x & 7) / 8.0, ub = (y & 7) / 8.0, uc = (z & 7) / 8.0; double q[3] = {0, 0, 0};
            for (int kk = 0; kk < 8; kk++) {
                int aa = a + (kk & 1), bb = b + (kk >> 1 & 1), cc = c + (kk >> 2); if (aa >= G) aa = G - 1; if (bb >= G) bb = G - 1; if (cc >= G) cc = G - 1;
                const double w = ((kk & 1) ? ua : 1 - ua) * ((kk >> 1 & 1) ? ub : 1 - ub) * ((kk >> 2) ? uc : 1 - uc);
                const double *l = lat + (((size_t)cc * G + bb) * G + aa) * 3; for (int d = 0; d < 3; d++) q[d] += w * l[d];
            }
            const double fx = q[0] - fo[2], fy = q[1] - fo[1], fz = q[2] - fo[0]; const int64_t ix = (int64_t)floor(fx), iy = (int64_t)floor(fy), iz = (int64_t)floor(fz);
            uint8_t *dst = in + ((size_t)z * T + y) * T + x; *dst = 0;
            if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= fn[2] || iy + 1 >= fn[1] || iz + 1 >= fn[0]) continue;
            const double ux = fx - ix, uy = fy - iy, uz = fz - iz; double sv = 0; int bad = 0;
            for (int kk = 0; kk < 8 && !bad; kk++) { const uint8_t v = fb[((size_t)(iz + (kk >> 2)) * fn[1] + iy + (kk >> 1 & 1)) * fn[2] + ix + (kk & 1)]; if (!v) bad = 1;
                sv += ((kk >> 2) ? uz : 1 - uz) * ((kk >> 1 & 1) ? uy : 1 - uy) * ((kk & 1) ? ux : 1 - ux) * v; }
            if (!bad) { *dst = (uint8_t)fmin(255, fmax(1, sv + 0.5)); nz++; }
        }
        free(fb);
        if (nz < j->minfrac * NT) continue;
        /* intensity pairs: target 4^3 block means vs the input at the block centre (both on the target grid) */
        for (int z = 0; z + 4 <= T; z += 4) for (int y = 0; y + 4 <= T; y += 4) for (int x = 0; x + 4 <= T; x += 4) {
            const uint8_t iv = in[((size_t)(z + 2) * T + y + 2) * T + x + 2]; if (!iv) continue;
            int sm = 0, ok = 1;
            for (int q = 0; q < 64 && ok; q++) { const uint8_t v = tg[((size_t)(z + (q >> 4)) * T + y + (q >> 2 & 3)) * T + x + (q & 3)]; if (!v) ok = 0; sm += v; }
            if (ok) h[(sm + 32) / 64][iv]++;
        }
        const int slot = atomic_fetch_add(&j->nacc, 1);
        if (slot >= j->want) break;
        memcpy(j->in + (size_t)slot * NT, in, NT); memcpy(j->tg + (size_t)slot * NT, tg, NT);
        memcpy(j->org[slot], o, sizeof o); j->tfrac[slot] = (float)nv / NT;
        if ((slot + 1) % 8 == 0) fprintf(stderr, "  sr-xpairs: %d / %d boxes (%d tried)\n", slot + 1, j->want, atomic_load(&j->next));
    }
    pthread_mutex_lock(&j->mu); for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) j->hist[a][b] += h[a][b]; pthread_mutex_unlock(&j->mu);
    free(in); free(tg); free(lat); free(h);
    return nullptr;
}
static rvol *x_vol(const char *spec, const char *cache, int *level, char *key, size_t kn) {   /* ROOT,KEY,UM,LEVEL */
    char buf[2048]; snprintf(buf, sizeof buf, "%s", spec ? spec : "");
    char *f[4] = {nullptr}; int n = 0; for (char *t = strtok(buf, ","); t && n < 4; t = strtok(nullptr, ",")) f[n++] = t;
    if (n != 4) { fprintf(stderr, "sr-xpairs: volume spec ROOT,KEY,UM,LEVEL\n"); return nullptr; }
    *level = atoi(f[3]); snprintf(key, kn, "%s", f[1]);
    rvol *v = rvol_open(f[0], f[1], cache, atof(f[2]));
    if (v && *level >= v->nl) { fprintf(stderr, "sr-xpairs: %s has no level %d\n", f[1], *level); return nullptr; }
    return v;
}
int cmd_sr_xpairs(int argc, char **argv) {
    const char *is = opt(argc, argv, "--input", nullptr), *ts = opt(argc, argv, "--target", nullptr), *cs = opt(argc, argv, "--chain", nullptr);
    if (argc < 3 || argv[2][0] == '-' || !is || !ts || !cs) {
        fprintf(stderr, "usage: ufsm sr-xpairs <out-dir> --input ROOT,KEY,UM,LEVEL --target ROOT,KEY,UM,LEVEL --chain MAP:c2f|f2c:PAIRDIR[;...] [--n 64] [--in 64]\n"
                        "       [--z Z0,Z1] [--holdout 0.15] [--minfrac 0.9] [--cover-mm 1.2] [--tol-um 2.4] [--threads 8] [--cache DIR] [--seed 1]\n");
        return 2;
    }
    const char *od = argv[2], *cache = opt(argc, argv, "--cache", nullptr);
    xjob *j = calloc(1, sizeof *j); char ikey[1024], tkey[1024];
    if (!(j->vi = x_vol(is, cache, &j->Li, ikey, sizeof ikey)) || !(j->vt = x_vol(ts, cache, &j->Lt, tkey, sizeof tkey))) return 1;
    j->same = atoi(opt(argc, argv, "--same-grid", "0")) != 0;
    j->B = atoi(opt(argc, argv, "--in", "64")); j->T = 4 * j->B;
    if (j->same) j->B = j->T = atoi(opt(argc, argv, "--box", "256")); j->want = atoi(opt(argc, argv, "--n", "64")); j->minfrac = atof(opt(argc, argv, "--minfrac", "0.9"));
    j->seed = (uint64_t)atoll(opt(argc, argv, "--seed", "1"));
    if (j->B < 8 || 512 % j->T || j->want < 1) { fprintf(stderr, "sr-xpairs: --in with 4 in dividing 512 (--same-grid: --box dividing 512)\n"); return 2; }
    char cb[4096]; snprintf(cb, sizeof cb, "%s", cs); char *save = nullptr;
    for (char *t = strtok_r(cb, ";", &save); t && j->ns < 4; t = strtok_r(nullptr, ";", &save)) {
        char *a = strtok(t, ":"), *d = strtok(nullptr, ":"), *pd = strtok(nullptr, ":");
        if (!a || !d || !pd) { fprintf(stderr, "sr-xpairs: --chain MAP:c2f|f2c:PAIRDIR\n"); return 2; }
        xstep *s = &j->st[j->ns]; rvol *cv = nullptr;
        if (reg_load_map(a, cache, &cv, nullptr, &s->m, &s->D)) return 1;
        if (s->D.d) s->m.D = &s->D;
        s->c2f = !strcmp(d, "c2f");
        if (rcover_load(&s->cov, pd, atof(opt(argc, argv, "--cover-mm", "1.2")) * 1000 / cv->um, atof(opt(argc, argv, "--tol-um", "2.4")) / cv->um)) return 1;
        rvol_close(cv); j->ns++;
    }
    j->z0 = 0; j->z1 = j->vi->shape[0][0];
    const char *zr = opt(argc, argv, "--z", nullptr);
    if (zr) { long long a, b; if (sscanf(zr, "%lld,%lld", &a, &b) != 2 || a < 0 || b <= a) { fprintf(stderr, "--z Z0,Z1\n"); return 2; } j->z0 = a; if (b < j->z1) j->z1 = b; }
    j->zcut = j->z1 - atof(opt(argc, argv, "--holdout", "0.15")) * (double)(j->z1 - j->z0);
    j->ntry = 400 * j->want;
    const size_t NB = (size_t)j->B * j->B * j->B, NT = (size_t)j->T * j->T * j->T;
    j->in = malloc((size_t)j->want * NB); j->tg = malloc((size_t)j->want * NT); j->org = calloc((size_t)j->want, sizeof *j->org); j->tfrac = calloc((size_t)j->want, sizeof *j->tfrac); j->split = calloc((size_t)j->want, 1);
    if (!j->in || !j->tg) { fprintf(stderr, "sr-xpairs: out of memory\n"); return 1; }
    pthread_mutex_init(&j->mu, nullptr); const time_t t0 = time(nullptr);
    const double ium = j->vi->um * (1 << j->Li), tum = j->vt->um * (1 << j->Lt);
    fprintf(stderr, "sr-xpairs: input %s level %d (%.3f um) %d^3 -> target %s level %d (%.3f um) at %.3f um, %d chain step(s), z %lld .. %lld (test from %.0f)\n",
            ikey, j->Li, ium, j->B, tkey, j->Lt, tum, ium / 4, j->ns, (long long)j->z0, (long long)j->z1, j->zcut);
    const int nt = atoi(opt(argc, argv, "--threads", "8")); pthread_t th[64]; const int TT = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < TT; t++) pthread_create(&th[t], nullptr, j->same ? x_worker_same : x_worker, j);
    for (int t = 0; t < TT; t++) pthread_join(th[t], nullptr);
    int n = atomic_load(&j->nacc); if (n > j->want) n = j->want;
    if (n < 1) { fprintf(stderr, "sr-xpairs: no box accepted\n"); return 1; }
    {   /* "test": the top --holdout fraction of the accepted boxes by z (the overlap with the target scan, not the whole input scan) */
        const double ho = atof(opt(argc, argv, "--holdout", "0.15")); int64_t *zs = malloc((size_t)n * sizeof *zs);
        for (int i = 0; i < n; i++) zs[i] = j->org[i][0];
        for (int a = 1; a < n; a++) for (int b = a; b > 0 && zs[b] < zs[b - 1]; b--) { const int64_t t = zs[b]; zs[b] = zs[b - 1]; zs[b - 1] = t; }
        const int nt_ = (int)lrint(ho * n); const int64_t zc = nt_ > 0 ? zs[n - nt_] : INT64_MAX;
        for (int i = 0; i < n; i++) j->split[i] = j->org[i][0] >= zc;
        j->zcut = nt_ > 0 ? (double)zc * (1 << j->Li) : 1e300; free(zs);
    }
    mkdir(od, 0755); char p[2200];
    snprintf(p, sizeof p, "%s/input.zarr", od); if (write_stack(p, j->in, n, j->B, "input (raw input-scan boxes)")) return 1;
    snprintf(p, sizeof p, "%s/target.zarr", od); if (write_stack(p, j->tg, n, j->T, "target (finer scan through the chain, x4 grid)")) return 1;
    snprintf(p, sizeof p, "%s/boxes.json", od); FILE *f = fopen(p, "w"); int ntest = 0;
    fprintf(f, "{\"source\":\"sr-xpairs\",\"input\":{\"key\":\"%s\",\"level\":%d,\"um\":%.6g,\"box\":%d%s},\"target\":{\"key\":\"%s\",\"level\":%d,\"um\":%.6g,\"grid_um\":%.6g,\"box\":%d,\"scale\":%d},\"ratio\":%.6g,\"chain\":\"%s\",\"n\":%d,\"requested\":%d,\"zcut_test\":%.1f,\n \"boxes\":[",
            ikey, j->Li, ium, j->B, j->same ? ",\"grid\":\"target\"" : "", tkey, j->Lt, tum, j->same ? tum : ium / 4, j->T, j->same ? 1 : 4, j->same ? ium / tum : 4.0, cs, n, j->want, j->zcut);
    for (int i = 0; i < n; i++) { ntest += j->split[i]; fprintf(f, "%s\n  {\"slot\":%d,\"origin_level_zyx\":[%lld,%lld,%lld],\"target_valid\":%.3f,\"split\":\"%s\"}", i ? "," : "", i,
                                        (long long)j->org[i][0], (long long)j->org[i][1], (long long)j->org[i][2], j->tfrac[i], j->split[i] ? "test" : "train"); }
    fprintf(f, "\n ],\"train\":%d,\"test\":%d}\n", n - ntest, ntest); fclose(f);
    double w[256], y[256], sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, tot = 0;
    for (int a = 0; a < 256; a++) { double c = 0, s = 0; for (int b = 1; b < 256; b++) { const double hc = (double)j->hist[a][b]; c += hc; s += b * hc; sx += a * hc; sy += b * hc; sxx += (double)a * a * hc; syy += (double)b * b * hc; sxy += (double)a * b * hc; } w[a] = c; y[a] = c > 0 ? s / c : 0; tot += c; }
    const double pr = tot > 1 ? (sxy / tot - sx / tot * sy / tot) / sqrt(fmax(1e-12, (sxx / tot - sx * sx / tot / tot) * (syy / tot - sy * sy / tot / tot))) : 0;
    double bw[256], bv[256]; int bs[256], nb = 0;
    for (int a = 0; a < 256; a++) { if (w[a] <= 0) continue; bv[nb] = y[a]; bw[nb] = w[a]; bs[nb] = a; nb++;
        while (nb > 1 && bv[nb - 2] > bv[nb - 1]) { bv[nb - 2] = (bv[nb - 2] * bw[nb - 2] + bv[nb - 1] * bw[nb - 1]) / (bw[nb - 2] + bw[nb - 1]); bw[nb - 2] += bw[nb - 1]; nb--; } }
    snprintf(p, sizeof p, "%s/intensity_map.json", od); f = fopen(p, "w");
    fprintf(f, "{\"maps\":\"target (4^3 block mean) -> input intensity\",\"method\":\"bin means + pool adjacent violators\",\"pearson_r\":%.4f,\"lut\":[", pr);
    int k2 = 0; for (int a = 0; a < 256; a++) { while (k2 + 1 < nb && bs[k2 + 1] <= a) k2++; fprintf(f, "%s%.3f", a ? "," : "", nb ? bv[k2] : (double)a); }
    fprintf(f, "],\"bin_counts\":["); for (int a = 0; a < 256; a++) fprintf(f, "%s%.0f", a ? "," : "", w[a]); fprintf(f, "]}\n"); fclose(f);
    fprintf(stderr, "sr-xpairs: %d boxes (%d train, %d test) from %d tries in %lds; intensity r %.3f -> %s\n", n, n - ntest, ntest, atomic_load(&j->next), (long)(time(nullptr) - t0), pr, od);
    return 0;
}

/* ufsm sr-resample <out.zarr> --input ROOT,KEY,UM,LEVEL --target ROOT,KEY,UM,LEVEL --chain MAP:c2f|f2c:PAIRDIR[;...] --shards FILE [--factor 4]
                    [--threads 8] [--cache DIR] [--q 1]
   The target scan resampled (trilinear, through the chain of registration maps as in ufsm sr-xpairs) onto the input scan's x factor grid
   (output voxel t at input-level coords (t + 0.5) / factor - 0.5): e.g. a real 2.4 um scan on the native 9.36 um scan's x4 grid, beside an SR
   prediction of that native scan. Only the listed 512^3 output shards ("sz sy sx" lines) are written (the array spans the input level x factor);
   0 where the target is masked or outside. Output: an OME group with one level named by its voxel size. */
#include "pyramid.h"
typedef struct { xjob *j; int64_t (*sh)[3]; int nsh; atomic_int next; z3w *w; int F; } rsjob;
static void *rs_worker(void *arg) {
    rsjob *r = arg; xjob *j = r->j; const int S = 512, G = S / 8 + 1, F = r->F; const double fi = (double)(1 << j->Li), ft = 1.0 / (1 << j->Lt);
    uint8_t *out = malloc((size_t)S * S * S); double *lat = malloc((size_t)G * G * G * 3 * sizeof *lat);
    for (;;) {
        const int k = atomic_fetch_add(&r->next, 1); if (k >= r->nsh) break;
        const int64_t *sh = r->sh[k]; memset(out, 0, (size_t)S * S * S);
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < G; c++) for (int b = 0; b < G; b++) for (int a = 0; a < G; a++) {
            const int64_t tv[3] = {sh[2] * S + 8 * a, sh[1] * S + 8 * b, sh[0] * S + 8 * c}; double p[3], f[3];   /* xyz output voxel */
            for (int d = 0; d < 3; d++) p[d] = (((tv[d] + 0.5) / F - 0.5) + 0.5) * fi - 0.5;
            x_map(j, p, f, nullptr);
            double *qq = lat + (((size_t)c * G + b) * G + a) * 3;
            for (int d = 0; d < 3; d++) { qq[d] = (f[d] + 0.5) * ft - 0.5; lo[d] = fmin(lo[d], qq[d]); hi[d] = fmax(hi[d], qq[d]); }
        }
        int64_t fo[3], fn[3]; for (int d = 0; d < 3; d++) { fo[2 - d] = (int64_t)floor(lo[d]) - 2; fn[2 - d] = (int64_t)ceil(hi[d]) + 2 - fo[2 - d] + 1; }
        int okr = (double)fn[0] * fn[1] * fn[2] <= 400e6;
        uint8_t *fb = okr ? malloc((size_t)fn[0] * fn[1] * fn[2]) : nullptr;
        if (fb && !rvol_read(j->vt, j->Lt, fo, fn, fb)) {
            for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
                const int a = x >> 3, b = y >> 3, c = z >> 3; const double ua = (x & 7) / 8.0, ub = (y & 7) / 8.0, uc = (z & 7) / 8.0; double q[3] = {0, 0, 0};
                for (int kk = 0; kk < 8; kk++) {
                    int aa = a + (kk & 1), bb = b + (kk >> 1 & 1), cc = c + (kk >> 2); if (aa >= G) aa = G - 1; if (bb >= G) bb = G - 1; if (cc >= G) cc = G - 1;
                    const double w = ((kk & 1) ? ua : 1 - ua) * ((kk >> 1 & 1) ? ub : 1 - ub) * ((kk >> 2) ? uc : 1 - uc);
                    const double *l = lat + (((size_t)cc * G + bb) * G + aa) * 3; for (int d = 0; d < 3; d++) q[d] += w * l[d];
                }
                const double fx = q[0] - fo[2], fy = q[1] - fo[1], fz = q[2] - fo[0]; const int64_t ix = (int64_t)floor(fx), iy = (int64_t)floor(fy), iz = (int64_t)floor(fz);
                if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= fn[2] || iy + 1 >= fn[1] || iz + 1 >= fn[0]) continue;
                const double ux = fx - ix, uy = fy - iy, uz = fz - iz; double sv = 0; int bad = 0;
                for (int kk = 0; kk < 8 && !bad; kk++) { const uint8_t v = fb[((size_t)(iz + (kk >> 2)) * fn[1] + iy + (kk >> 1 & 1)) * fn[2] + ix + (kk & 1)]; if (!v) bad = 1;
                    sv += ((kk >> 2) ? uz : 1 - uz) * ((kk >> 1 & 1) ? uy : 1 - uy) * ((kk & 1) ? ux : 1 - ux) * v; }
                if (!bad) out[((size_t)z * S + y) * S + x] = (uint8_t)fmin(255, fmax(1, sv + 0.5));
            }
        }
        free(fb);
        if (z3w_write_shard(r->w, sh[0], sh[1], sh[2], out, 1)) fprintf(stderr, "sr-resample: %s\n", z3w_error());
        fprintf(stderr, "\rsr-resample: shard %d / %d   ", k + 1, r->nsh);
    }
    free(out); free(lat); return nullptr;
}
int cmd_sr_resample(int argc, char **argv) {
    const char *is = opt(argc, argv, "--input", nullptr), *ts = opt(argc, argv, "--target", nullptr), *cs = opt(argc, argv, "--chain", nullptr), *sf = opt(argc, argv, "--shards", nullptr);
    if (argc < 3 || argv[2][0] == '-' || !is || !ts || !cs || !sf) { fprintf(stderr, "usage: ufsm sr-resample <out.zarr> --input ROOT,KEY,UM,LEVEL --target ROOT,KEY,UM,LEVEL --chain MAP:c2f|f2c:PAIRDIR[;...] --shards FILE [--factor 4] [--threads 8] [--cache DIR] [--q 1]\n"); return 2; }
    const char *cache = opt(argc, argv, "--cache", nullptr); xjob *j = calloc(1, sizeof *j); char ikey[1024], tkey[1024];
    if (!(j->vi = x_vol(is, cache, &j->Li, ikey, sizeof ikey)) || !(j->vt = x_vol(ts, cache, &j->Lt, tkey, sizeof tkey))) return 1;
    char cb[4096]; snprintf(cb, sizeof cb, "%s", cs); char *save = nullptr;
    for (char *t = strtok_r(cb, ";", &save); t && j->ns < 4; t = strtok_r(nullptr, ";", &save)) {
        char *a = strtok(t, ":"), *d = strtok(nullptr, ":"), *pd = strtok(nullptr, ":"); if (!a || !d || !pd) return 2;
        xstep *s = &j->st[j->ns]; if (reg_load_map(a, cache, nullptr, nullptr, &s->m, &s->D)) return 1;
        if (s->D.d) s->m.D = &s->D; s->c2f = !strcmp(d, "c2f"); j->ns++;
    }
    const int F = atoi(opt(argc, argv, "--factor", "4")); const double oum = j->vi->um * (1 << j->Li) / F;
    rsjob r = {0}; r.j = j; r.F = F;
    FILE *f = fopen(sf, "r"); if (!f) return 1; long long a, b, c; int cap = 0;
    while (fscanf(f, "%lld %lld %lld", &a, &b, &c) == 3) { if (r.nsh == cap) r.sh = realloc(r.sh, (size_t)(cap = cap ? 2 * cap : 256) * sizeof *r.sh); r.sh[r.nsh][0] = a; r.sh[r.nsh][1] = b; r.sh[r.nsh][2] = c; r.nsh++; }
    fclose(f);
    int64_t shape[3]; for (int d = 0; d < 3; d++) shape[d] = j->vi->shape[j->Li][d] * F;
    char lv[32], ld[2200]; z3w_level_name(oum, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", argv[2], lv);
    char attrs[2600]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"sr-resample: %s level %d on %s level %d x%d grid\",\"chain\":\"%s\"}}", tkey, j->Lt, ikey, j->Li, F, cs);
    r.w = z3w_create(ld, shape, 512, (float)atof(opt(argc, argv, "--q", "1")), 0, attrs); if (!r.w) { fprintf(stderr, "sr-resample: %s\n", z3w_error()); return 1; }
    const int nt = atoi(opt(argc, argv, "--threads", "8")); pthread_t th[64]; const int TT = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < TT; t++) pthread_create(&th[t], nullptr, rs_worker, &r);
    for (int t = 0; t < TT; t++) pthread_join(th[t], nullptr);
    fprintf(stderr, "\n");
    if (z3w_close(r.w) || pyramid_write_group(argv[2], oum, 1, "ufsm-sr-resample", attrs)) return 1;
    fprintf(stderr, "sr-resample: %d shards -> %s (%.4g um)\n", r.nsh, argv[2], oum);
    return 0;
}
